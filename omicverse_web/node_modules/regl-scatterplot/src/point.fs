const FRAGMENT_SHADER = `
precision highp float;

uniform float antiAliasing;

varying vec4 color;
varying float finalPointSize;

float linearstep(float edge0, float edge1, float x) {
  return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float sdf = length(c) * finalPointSize;
  float alpha = linearstep(finalPointSize + antiAliasing, finalPointSize - antiAliasing, sdf);

  gl_FragColor = vec4(color.rgb, alpha * color.a);
}
`;

export default FRAGMENT_SHADER;
