/**
 * Basic 3x3 matrix operations for 2D transformations
 * Simplified version of gl-matrix functionality
 */

export const mat3 = {
    // Create a new 3x3 identity matrix
    create() {
        return new Float32Array([
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ]);
    },

    // Set matrix values
    set(out, m00, m01, m02, m10, m11, m12, m20, m21, m22) {
        out[0] = m00;
        out[1] = m01;
        out[2] = m02;
        out[3] = m10;
        out[4] = m11;
        out[5] = m12;
        out[6] = m20;
        out[7] = m21;
        out[8] = m22;
        return out;
    },

    // Copy matrix
    copy(out, a) {
        out[0] = a[0];
        out[1] = a[1];
        out[2] = a[2];
        out[3] = a[3];
        out[4] = a[4];
        out[5] = a[5];
        out[6] = a[6];
        out[7] = a[7];
        out[8] = a[8];
        return out;
    },

    // Set to identity matrix
    identity(out) {
        out[0] = 1;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 1;
        out[5] = 0;
        out[6] = 0;
        out[7] = 0;
        out[8] = 1;
        return out;
    },

    // Matrix multiplication: out = a * b
    multiply(out, a, b) {
        const a00 = a[0], a01 = a[1], a02 = a[2];
        const a10 = a[3], a11 = a[4], a12 = a[5];
        const a20 = a[6], a21 = a[7], a22 = a[8];

        const b00 = b[0], b01 = b[1], b02 = b[2];
        const b10 = b[3], b11 = b[4], b12 = b[5];
        const b20 = b[6], b21 = b[7], b22 = b[8];

        out[0] = b00 * a00 + b01 * a10 + b02 * a20;
        out[1] = b00 * a01 + b01 * a11 + b02 * a21;
        out[2] = b00 * a02 + b01 * a12 + b02 * a22;

        out[3] = b10 * a00 + b11 * a10 + b12 * a20;
        out[4] = b10 * a01 + b11 * a11 + b12 * a21;
        out[5] = b10 * a02 + b11 * a12 + b12 * a22;

        out[6] = b20 * a00 + b21 * a10 + b22 * a20;
        out[7] = b20 * a01 + b21 * a11 + b22 * a21;
        out[8] = b20 * a02 + b21 * a12 + b22 * a22;

        return out;
    },

    // Create translation matrix
    translation(out, x, y) {
        out[0] = 1;
        out[1] = 0;
        out[2] = x;
        out[3] = 0;
        out[4] = 1;
        out[5] = y;
        out[6] = 0;
        out[7] = 0;
        out[8] = 1;
        return out;
    },

    // Create scaling matrix
    scaling(out, x, y) {
        out[0] = x;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = y;
        out[5] = 0;
        out[6] = 0;
        out[7] = 0;
        out[8] = 1;
        return out;
    },

    // Create rotation matrix
    rotation(out, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        out[0] = cos;
        out[1] = -sin;
        out[2] = 0;
        out[3] = sin;
        out[4] = cos;
        out[5] = 0;
        out[6] = 0;
        out[7] = 0;
        out[8] = 1;
        return out;
    },

    // Translate matrix
    translate(out, a, x, y) {
        const temp = this.create();
        this.translation(temp, x, y);
        return this.multiply(out, temp, a);
    },

    // Scale matrix
    scale(out, a, x, y) {
        const temp = this.create();
        this.scaling(temp, x, y);
        return this.multiply(out, temp, a);
    },

    // Rotate matrix
    rotate(out, a, angle) {
        const temp = this.create();
        this.rotation(temp, angle);
        return this.multiply(out, temp, a);
    },

    // Transform a 2D point
    transformPoint(out, mat, point) {
        const x = point[0];
        const y = point[1];

        out[0] = mat[0] * x + mat[1] * y + mat[2];
        out[1] = mat[3] * x + mat[4] * y + mat[5];

        return out;
    },

    // Invert matrix (simplified for 2D transformations)
    invert(out, a) {
        const a00 = a[0], a01 = a[1], a02 = a[2];
        const a10 = a[3], a11 = a[4], a12 = a[5];
        const a20 = a[6], a21 = a[7], a22 = a[8];

        const b01 = a22 * a11 - a12 * a21;
        const b11 = -a22 * a10 + a12 * a20;
        const b21 = a21 * a10 - a11 * a20;

        // Calculate the determinant
        let det = a00 * b01 + a01 * b11 + a02 * b21;

        if (!det) {
            return null;
        }
        det = 1.0 / det;

        out[0] = b01 * det;
        out[1] = (-a22 * a01 + a02 * a21) * det;
        out[2] = (a12 * a01 - a02 * a11) * det;
        out[3] = b11 * det;
        out[4] = (a22 * a00 - a02 * a20) * det;
        out[5] = (-a12 * a00 + a02 * a10) * det;
        out[6] = b21 * det;
        out[7] = (-a21 * a00 + a01 * a20) * det;
        out[8] = (a11 * a00 - a01 * a10) * det;

        return out;
    }
};