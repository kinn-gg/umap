package umap

import "fmt"

type ValidationError struct{ Message string }

func (e *ValidationError) Error() string { return e.Message }

type ShapeError struct{ Message string }

func (e *ShapeError) Error() string { return e.Message }

type NumericError struct{ Message string }

func (e *NumericError) Error() string { return e.Message }

type UnsupportedError struct{ Message string }

func (e *UnsupportedError) Error() string { return e.Message }

func validationf(format string, args ...any) error {
	return &ValidationError{fmt.Sprintf(format, args...)}
}
func shapef(format string, args ...any) error   { return &ShapeError{fmt.Sprintf(format, args...)} }
func numericf(format string, args ...any) error { return &NumericError{fmt.Sprintf(format, args...)} }
func unsupportedf(format string, args ...any) error {
	return &UnsupportedError{fmt.Sprintf(format, args...)}
}
