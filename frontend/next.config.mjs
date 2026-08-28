/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow the platform's live-preview origin and localhost when running the dev server.
  allowedDevOrigins: ['localhost', '127.0.0.1', '*.e2b.app'],
};

export default nextConfig;
