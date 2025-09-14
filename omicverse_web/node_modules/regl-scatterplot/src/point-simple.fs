const FRAGMENT_SHADER = `precision highp float;

varying vec4 color;

void main() {
  gl_FragColor = color;
}
`;

export default FRAGMENT_SHADER;
