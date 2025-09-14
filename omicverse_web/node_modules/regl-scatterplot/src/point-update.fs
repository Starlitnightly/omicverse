const SHADER = `precision highp float;

uniform sampler2D startStateTex;
uniform sampler2D endStateTex;
uniform float t;

varying vec2 particleTextureIndex;

void main() {
  // Interpolate x, y, and value
  vec3 start = texture2D(startStateTex, particleTextureIndex).xyw;
  vec3 end = texture2D(endStateTex, particleTextureIndex).xyw;
  vec3 curr = start * (1.0 - t) + end * t;

  // The category cannot be interpolated
  float endCategory = texture2D(endStateTex, particleTextureIndex).z;

  gl_FragColor = vec4(curr.xy, endCategory, curr.z);
}`;

export default SHADER;
