package parquet_test

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/google/uuid"
	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/variant"
)

// This file property-tests the shredded write→read round trip over
// randomly generated shredding schemas and randomly generated variant
// values. Schemas and values draw field names and primitive types from the
// same small pools, so every relationship of the VariantShredding.md case
// table arises statistically: exact matches into typed_value, type
// mismatches falling back to value, partially shredded objects with
// residual fields, missing fields, empty objects and arrays, variant
// nulls, and deeply nested combinations of all of these.
//
// The property: for any shredding schema S and any variant value v,
// writing v through ShreddedVariant(S) and reading it back yields a value
// structurally equal to v. Values enter and leave as raw variant binary,
// so every variant type participates, not just the ones with Go-native
// mappings.

// shredFieldNames is the shared object field name pool. Values also use an
// out-of-pool name so residual (never-shredded) fields occur.
var shredFieldNames = [...]string{"a", "b", "c", "d"}

// randomShredNode generates a random typed_value schema node covering the
// full shredded-types table plus LIST and object groups.
func randomShredNode(r *rand.Rand, depth int) parquet.Node {
	if depth < 2 {
		switch r.IntN(8) {
		case 0:
			return parquet.List(randomShredNode(r, depth+1))
		case 1, 2:
			g := parquet.Group{}
			for _, name := range shredFieldNames {
				if r.IntN(2) == 0 {
					g[name] = randomShredNode(r, depth+1)
				}
			}
			if len(g) == 0 { // empty object typed_value groups are invalid
				g[shredFieldNames[r.IntN(len(shredFieldNames))]] = randomShredNode(r, depth+1)
			}
			return g
		}
	}
	switch r.IntN(17) {
	case 0:
		return parquet.Leaf(parquet.BooleanType)
	case 1:
		return parquet.Int(8)
	case 2:
		return parquet.Int(16)
	case 3:
		return parquet.Int(32)
	case 4:
		return parquet.Int(64)
	case 5:
		return parquet.Leaf(parquet.FloatType)
	case 6:
		return parquet.Leaf(parquet.DoubleType)
	case 7:
		return parquet.String()
	case 8:
		return parquet.Leaf(parquet.ByteArrayType)
	case 9:
		return parquet.Date()
	case 10:
		return parquet.UUID()
	case 11:
		return parquet.TimestampAdjusted(parquet.Microsecond, r.IntN(2) == 0)
	case 12:
		return parquet.TimestampAdjusted(parquet.Nanosecond, r.IntN(2) == 0)
	case 13:
		return parquet.TimeAdjusted(parquet.Microsecond, false) // spec: TIME(false, MICROS)
	case 14:
		return parquet.Decimal(2, 9, parquet.Int32Type)
	case 15:
		return parquet.Decimal(2, 18, parquet.Int64Type)
	default:
		return parquet.Decimal(2, 38, parquet.FixedLenByteArrayType(16))
	}
}

// randomVariant generates a random variant value from the same pools, so
// it sometimes matches a generated schema exactly, sometimes partially,
// and sometimes not at all.
func randomVariant(r *rand.Rand, depth int) variant.Value {
	if depth < 3 {
		switch r.IntN(6) {
		case 0:
			elems := make([]variant.Value, r.IntN(4))
			for i := range elems {
				elems[i] = randomVariant(r, depth+1)
			}
			return variant.MakeArray(elems)
		case 1:
			var fields []variant.Field
			for _, name := range shredFieldNames {
				if r.IntN(2) == 0 {
					fields = append(fields, variant.Field{Name: name, Value: randomVariant(r, depth+1)})
				}
			}
			if r.IntN(4) == 0 {
				fields = append(fields, variant.Field{Name: "resid", Value: randomVariant(r, depth+1)})
			}
			return variant.MakeObject(fields)
		}
	}
	switch r.IntN(17) {
	case 0:
		return variant.Null()
	case 1:
		return variant.Bool(r.IntN(2) == 0)
	case 2:
		return variant.Int8(int8(r.IntN(1 << 8)))
	case 3:
		return variant.Int16(int16(r.IntN(1 << 16)))
	case 4:
		return variant.Int32(int32(r.Uint32()))
	case 5:
		return variant.Int64(int64(r.Uint64()))
	case 6:
		return variant.Float(float32(r.NormFloat64()))
	case 7:
		return variant.Double(r.NormFloat64())
	case 8:
		n := r.IntN(80) // crosses the 63-byte short-string boundary
		b := make([]byte, n)
		for i := range b {
			b[i] = byte('a' + r.IntN(26))
		}
		return variant.String(string(b))
	case 9:
		b := make([]byte, r.IntN(20))
		for i := range b {
			b[i] = byte(r.Uint32())
		}
		return variant.Binary(b)
	case 10:
		return variant.Date(int32(r.IntN(40000)))
	case 11:
		var u uuid.UUID
		for i := range u {
			u[i] = byte(r.Uint32())
		}
		return variant.UUID(u)
	case 12:
		ts := int64(r.Uint64() >> 20)
		switch r.IntN(4) {
		case 0:
			return variant.Timestamp(ts)
		case 1:
			return variant.TimestampNTZ(ts)
		case 2:
			return variant.TimestampNanos(ts)
		default:
			return variant.TimestampNTZNanos(ts)
		}
	case 13:
		return variant.Time(int64(r.IntN(86400_000_000)))
	case 14:
		// Scale 2 matches the generated decimal columns; scale 3 must
		// fall back to the value column.
		return variant.Decimal4(int32(r.Uint32()), byte(2+r.IntN(2)))
	case 15:
		return variant.Decimal8(int64(r.Uint64()), byte(2+r.IntN(2)))
	default:
		var d [16]byte
		for i := range d {
			d[i] = byte(r.Uint32())
		}
		return variant.Decimal16(d, byte(2+r.IntN(2)))
	}
}

func encodeRawVariant(v variant.Value) rawVariant {
	var b variant.MetadataBuilder
	value := variant.Encode(&b, v)
	_, metadata := b.Build()
	return rawVariant{Metadata: metadata, Value: value}
}

// shreddedRoundTrip writes the given variant values through a shredded
// variant column with the given typed_value schema and asserts that every
// row reads back structurally equal.
func shreddedRoundTrip(t *testing.T, shred parquet.Node, values []variant.Value) {
	t.Helper()
	shredded, err := parquet.ShreddedVariant(shred)
	if err != nil {
		t.Fatalf("building shredded variant node: %v", err)
	}
	schema := parquet.NewSchema("table", parquet.Group{
		"id":  parquet.Int(32),
		"var": shredded,
	})

	type row struct {
		ID  int32 `parquet:"id"`
		Var any   `parquet:"var,variant"`
	}
	rows := make([]row, len(values))
	for i, v := range values {
		rows[i] = row{ID: int32(i), Var: encodeRawVariant(v)}
	}

	buf := new(bytes.Buffer)
	w := parquet.NewGenericWriter[row](buf, schema)
	if _, err := w.Write(rows); err != nil {
		t.Fatalf("schema:\n%s\nwriting: %v", schema, err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("schema:\n%s\nclosing: %v", schema, err)
	}

	got, err := parquet.Read[rawVariantRow](bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("schema:\n%s\nreading: %v", schema, err)
	}
	if len(got) != len(values) {
		t.Fatalf("schema:\n%s\nread %d rows, want %d", schema, len(got), len(values))
	}
	for i, want := range values {
		decoded, err := decodeRawVariant(got[i].Var)
		if err != nil {
			t.Errorf("schema:\n%s\nrow %d (%#v): decoding read-back value: %v", schema, i, want.GoValue(), err)
			continue
		}
		if !decoded.Equal(want) {
			t.Errorf("schema:\n%s\nrow %d round trip mismatch:\n got: %#v\nwant: %#v",
				schema, i, decoded.GoValue(), want.GoValue())
		}
	}
}

func TestShreddedVariantRandomizedRoundTrip(t *testing.T) {
	const numSchemas, rowsPerSchema = 64, 8
	r := rand.New(rand.NewPCG(0, 1))
	for i := range numSchemas {
		t.Run(fmt.Sprintf("schema_%02d", i), func(t *testing.T) {
			shred := randomShredNode(r, 0)
			values := make([]variant.Value, rowsPerSchema)
			for j := range values {
				values[j] = randomVariant(r, 0)
			}
			shreddedRoundTrip(t, shred, values)
		})
	}
}

// FuzzShreddedVariantRoundTrip lets the fuzzer explore the same property
// over arbitrary generator seeds.
func FuzzShreddedVariantRoundTrip(f *testing.F) {
	for seed := range uint64(8) {
		f.Add(seed, seed*2654435761)
	}
	f.Fuzz(func(t *testing.T, s1, s2 uint64) {
		r := rand.New(rand.NewPCG(s1, s2))
		shred := randomShredNode(r, 0)
		values := make([]variant.Value, 4)
		for i := range values {
			values[i] = randomVariant(r, 0)
		}
		shreddedRoundTrip(t, shred, values)
	})
}
