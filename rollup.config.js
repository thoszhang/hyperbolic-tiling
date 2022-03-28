import typescript from "@rollup/plugin-typescript";
import { nodeResolve } from "@rollup/plugin-node-resolve";

export default {
  input: "src/script.ts",
  output: {
    file: "dist/script.js",
    format: "cjs",
    sourcemap: "inline",
  },
  plugins: [nodeResolve(), typescript()],
};
