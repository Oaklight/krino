import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/krino",
  images: { unoptimized: true },
};

export default nextConfig;
