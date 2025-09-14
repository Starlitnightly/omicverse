const FRAGMENT_SHADER = `
precision mediump float;

uniform sampler2D texture;

varying vec2 uv;

void main () {
  gl_FragColor = texture2D(texture, uv);
}
`;

export default FRAGMENT_SHADER;
