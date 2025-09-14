"""
Matrix FlatBuffers serialization module
Based on CellxGene's high-performance data serialization
"""

import json
import numpy as np
import pandas as pd
from flatbuffers import Builder
from scipy import sparse

# Import generated FlatBuffers modules (module-level functions) and classes
from .NetEncoding import Column as ColumnModule
from .NetEncoding import Matrix as MatrixModule
from .NetEncoding import Float32Array as Float32ArrayModule
from .NetEncoding import Float64Array as Float64ArrayModule
from .NetEncoding import Int32Array as Int32ArrayModule
from .NetEncoding import Uint32Array as Uint32ArrayModule
from .NetEncoding import JSONEncodedArray as JSONEncodedArrayModule
from .NetEncoding.TypedArray import TypedArray


def get_encoding_dtype_of_array(arr):
    """
    Determine the best encoding type for array data
    """
    if isinstance(arr, (pd.Series, pd.Index)):
        if pd.api.types.is_numeric_dtype(arr):
            dtype = arr.dtype
        else:
            return "object"  # Will be JSON encoded
    elif isinstance(arr, np.ndarray):
        dtype = arr.dtype
    else:
        return "object"

    # Map numpy dtypes to encoding types
    if np.issubdtype(dtype, np.integer):
        if dtype.itemsize <= 4:
            return "int32"
        else:
            return "int32"  # Downcast large integers
    elif np.issubdtype(dtype, np.floating):
        return "float32"  # Use float32 for efficiency
    else:
        return "object"


def serialize_typed_array(builder, source_array, encoding_info):
    """
    Serialize array data into FlatBuffers typed array format
    """
    arr = source_array
    array_type, as_type = encoding_info(source_array)

    if isinstance(arr, pd.Index):
        arr = arr.to_series()

    # Convert to numpy array
    if as_type == "json":
        as_json = arr.to_json(orient="records")
        arr = np.array(bytearray(as_json, "utf-8"))
    else:
        if sparse.issparse(arr):
            arr = arr.toarray()
        elif isinstance(arr, pd.Series):
            arr = arr.to_numpy()
        if arr.dtype != as_type:
            arr = arr.astype(as_type)

    # Handle 2D arrays
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr.T[0]

    # Create FlatBuffers vector
    vec = builder.CreateNumpyVector(arr)

    # Serialize typed array table
    if array_type == TypedArray.Int32Array:
        Int32ArrayModule.Start(builder)
        Int32ArrayModule.AddData(builder, vec)
        array_value = Int32ArrayModule.End(builder)
    elif array_type == TypedArray.Uint32Array:
        Uint32ArrayModule.Start(builder)
        Uint32ArrayModule.AddData(builder, vec)
        array_value = Uint32ArrayModule.End(builder)
    elif array_type == TypedArray.Float32Array:
        Float32ArrayModule.Start(builder)
        Float32ArrayModule.AddData(builder, vec)
        array_value = Float32ArrayModule.End(builder)
    elif array_type == TypedArray.Float64Array:
        Float64ArrayModule.Start(builder)
        Float64ArrayModule.AddData(builder, vec)
        array_value = Float64ArrayModule.End(builder)
    elif array_type == TypedArray.JSONEncodedArray:
        JSONEncodedArrayModule.Start(builder)
        JSONEncodedArrayModule.AddData(builder, vec)
        array_value = JSONEncodedArrayModule.End(builder)
    else:
        raise ValueError(f"Unsupported array type: {array_type}")

    return array_type, array_value


def serialize_column(builder, typed_arr):
    """Serialize Column"""
    u_type, u_value = typed_arr
    ColumnModule.Start(builder)
    ColumnModule.AddUType(builder, u_type)
    ColumnModule.AddU(builder, u_value)
    return ColumnModule.End(builder)


def serialize_matrix(builder, n_rows, n_cols, columns, col_idx):
    """Serialize Matrix"""
    MatrixModule.Start(builder)
    MatrixModule.AddNRows(builder, n_rows)
    MatrixModule.AddNCols(builder, n_cols)
    MatrixModule.AddColumns(builder, columns)
    if col_idx is not None:
        u_type, u_val = col_idx
        MatrixModule.AddColIndexType(builder, u_type)
        MatrixModule.AddColIndex(builder, u_val)
    return MatrixModule.End(builder)


def column_encoding(arr):
    """Determine column encoding type"""
    column_encoding_type_map = {
        np.dtype(np.float64).str: (TypedArray.Float32Array, np.float32),
        np.dtype(np.float32).str: (TypedArray.Float32Array, np.float32),
        np.dtype(np.float16).str: (TypedArray.Float32Array, np.float32),
        np.dtype(np.int8).str: (TypedArray.Int32Array, np.int32),
        np.dtype(np.int16).str: (TypedArray.Int32Array, np.int32),
        np.dtype(np.int32).str: (TypedArray.Int32Array, np.int32),
        np.dtype(np.int64).str: (TypedArray.Int32Array, np.int32),
        np.dtype(np.uint8).str: (TypedArray.Uint32Array, np.uint32),
        np.dtype(np.uint16).str: (TypedArray.Uint32Array, np.uint32),
        np.dtype(np.uint32).str: (TypedArray.Uint32Array, np.uint32),
        np.dtype(np.uint64).str: (TypedArray.Uint32Array, np.uint32),
    }
    column_encoding_default = (TypedArray.JSONEncodedArray, "json")

    encoding_dtype = np.dtype(get_encoding_dtype_of_array(arr))
    return column_encoding_type_map.get(encoding_dtype.str, column_encoding_default)


def index_encoding(arr):
    """Determine index encoding type"""
    index_encoding_type_map = {
        np.dtype(np.int32).str: (TypedArray.Int32Array, np.int32),
        np.dtype(np.int64).str: (TypedArray.Int32Array, np.int32),
        np.dtype(np.uint32).str: (TypedArray.Uint32Array, np.uint32),
        np.dtype(np.uint64).str: (TypedArray.Uint32Array, np.uint32),
    }
    index_encoding_default = (TypedArray.JSONEncodedArray, "json")

    return index_encoding_type_map.get(arr.dtype.str, index_encoding_default)


def guess_at_mem_needed(matrix):
    """Estimate memory needed for serialization"""
    n_rows, n_cols = matrix.shape
    if isinstance(matrix, np.ndarray) or sparse.issparse(matrix):
        guess = (n_rows * n_cols * matrix.dtype.itemsize) + 1024
    elif isinstance(matrix, pd.DataFrame):
        # Estimate DataFrame size
        guess = n_rows * n_cols * 8 + 1024
    else:
        guess = 1024

    # Round up to nearest 1024 bytes
    guess = (guess + 0x400) & (~0x3FF)
    return guess


def encode_matrix_fbs(matrix, row_idx=None, col_idx=None):
    """
    Encode 2D matrix data into FlatBuffers format for high-performance transmission

    Args:
        matrix: 2D DataFrame, ndarray or sparse matrix
        row_idx: Row index (currently unsupported, must be None)
        col_idx: Column index

    Returns:
        bytes: FlatBuffers encoded data
    """
    if row_idx is not None:
        raise ValueError("Row indexing not supported for FBS Matrix")
    if matrix.ndim != 2:
        raise ValueError("FBS Matrix must be 2D")

    n_rows, n_cols = matrix.shape

    # Estimate buffer size
    builder = Builder(guess_at_mem_needed(matrix))

    # Serialize columns (in reverse order for FlatBuffers)
    columns = []
    for cidx in range(n_cols - 1, -1, -1):
        if isinstance(matrix, pd.DataFrame):
            col = matrix.iloc[:, cidx]
        else:
            col = matrix[:, cidx]

        typed_arr = serialize_typed_array(builder, col, column_encoding)
        columns.append(serialize_column(builder, typed_arr))

    # Create columns vector
    MatrixModule.StartColumnsVector(builder, n_cols)
    for c in columns:
        builder.PrependUOffsetTRelative(c)
    matrix_column_vec = builder.EndVector(n_cols)

    # Serialize column index if provided
    cidx = None
    if col_idx is not None:
        cidx = serialize_typed_array(builder, col_idx, index_encoding)

    # Create matrix
    matrix = serialize_matrix(builder, n_rows, n_cols, matrix_column_vec, cidx)

    builder.Finish(matrix)
    return builder.Output()


def deserialize_typed_array(tarr):
    """Deserialize typed array from FlatBuffers"""
    type_map = {
        TypedArray.NONE: None,
        TypedArray.Uint32Array: Uint32ArrayModule.Uint32Array,
        TypedArray.Int32Array: Int32ArrayModule.Int32Array,
        TypedArray.Float32Array: Float32ArrayModule.Float32Array,
        TypedArray.Float64Array: Float64ArrayModule.Float64Array,
        TypedArray.JSONEncodedArray: JSONEncodedArrayModule.JSONEncodedArray,
    }

    u_type, u = tarr
    if u_type == TypedArray.NONE:
        return None

    TarType = type_map.get(u_type)
    if TarType is None:
        raise TypeError(f"FBS contains unknown data type: {u_type}")

    arr = TarType()
    arr.Init(u.Bytes, u.Pos)
    narr = arr.DataAsNumpy()

    if u_type == TypedArray.JSONEncodedArray:
        narr = json.loads(narr.tobytes().decode("utf-8"))

    return narr


def decode_matrix_fbs(fbs):
    """
    Decode FlatBuffers matrix data back to DataFrame

    Args:
        fbs: FlatBuffers encoded bytes

    Returns:
        pd.DataFrame: Decoded matrix data
    """
    matrix = MatrixModule.Matrix.GetRootAs(fbs, 0)
    n_rows = matrix.NRows()
    n_cols = matrix.NCols()

    if n_rows == 0 or n_cols == 0:
        return pd.DataFrame()

    # Current schema does not carry a row index union; only check column index

    columns_length = matrix.ColumnsLength()

    # Get column index
    columns_index = deserialize_typed_array((matrix.ColIndexType(), matrix.ColIndex()))
    if columns_index is None:
        columns_index = range(0, n_cols)

    # Sanity checks
    if len(columns_index) != n_cols or columns_length != n_cols:
        raise ValueError("FBS column count does not match number of columns in underlying matrix")

    # Deserialize column data
    columns_data = {}
    columns_type = {}
    for col_idx in range(0, columns_length):
        col = matrix.Columns(col_idx)
        tarr = (col.UType(), col.U())
        data = deserialize_typed_array(tarr)
        columns_data[columns_index[col_idx]] = data

        if len(data) != n_rows:
            raise ValueError("FBS column length does not match number of rows")

        if col.UType() == TypedArray.JSONEncodedArray:
            columns_type[columns_index[col_idx]] = "category"

    # Create DataFrame
    df = pd.DataFrame.from_dict(data=columns_data).astype(columns_type, copy=False)

    # Final sanity checks
    if not df.columns.is_unique or len(df.columns) != n_cols:
        raise KeyError("FBS column indices are not unique")

    return df
