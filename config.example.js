// Copy this file to config.js and fill in your values
// This file is safe to commit to public repo

export const DEPLOY_CONFIG = {
  // GitHub Pages URL
  siteUrl: 'https://tandat8896.github.io',
  
  // Build settings
  nodeEnv: 'production',
  siteName: 'Tấn Đạt Blog',
  
  // Analytics (optional)
  // gaTrackingId: 'your-ga-id',
  // gtmId: 'your-gtm-id',
  
  // CI/CD settings (will be overridden by environment)
  deployBranch: process.env.DEPLOY_BRANCH || 'main',
  buildCommand: process.env.BUILD_COMMAND || 'pnpm run build',
};
