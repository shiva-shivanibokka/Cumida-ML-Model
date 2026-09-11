/** @type {import('next').NextConfig} */
const nextConfig = {
  // A folder of static files: no server, no revalidation, nothing to keep warm.
  // The models run in the visitor's browser, so there is nothing for a backend
  // to do even if one existed.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
