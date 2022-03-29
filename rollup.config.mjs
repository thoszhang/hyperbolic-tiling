import typescript from "@rollup/plugin-typescript";
import { nodeResolve } from "@rollup/plugin-node-resolve";
import { terser } from "rollup-plugin-terser";
import glslOptimize from "rollup-plugin-glsl-optimize";

const production = !process.env.ROLLUP_WATCH;

export default {
  input: "src/script.ts",
  output: {
    file: "dist/script.js",
    format: "cjs",
    sourcemap: !production && "inline",
  },
  plugins: [
    nodeResolve(),
    typescript({ sourceMap: !production, inlineSources: !production }),
    glslOptimize({ include: "src/*.glsl" }),
    production && terser(),
  ],
};
