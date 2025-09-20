# Block Components Architecture

This directory contains modular block components that make up the AstroPaper blog theme. Each block is designed to be reusable, maintainable, and easily customizable.

## Available Block Components

### NavigationBlock.astro
- **Purpose**: Header navigation with logo, menu items, and theme toggle
- **Features**: Responsive design, mobile menu, active state highlighting
- **Props**: None (uses SITE config)

### HeroBlock.astro
- **Purpose**: Hero section with title, description, and social links
- **Props**:
  - `title?: string` - Main title (default: "Mingalaba")
  - `subtitle?: string` - Optional subtitle
  - `description?: string` - Description text
  - `showRss?: boolean` - Show RSS link (default: true)
  - `showSocials?: boolean` - Show social links (default: true)

### FeaturedPostsBlock.astro
- **Purpose**: Showcase featured blog posts
- **Props**:
  - `posts: CollectionEntry<"blog">[]` - Array of featured posts
  - `showDivider?: boolean` - Show divider below (default: true)

### RecentPostsBlock.astro
- **Purpose**: Display recent blog posts with pagination
- **Props**:
  - `posts: CollectionEntry<"blog">[]` - Array of recent posts
  - `showViewAll?: boolean` - Show "View All" button (default: true)
  - `maxPosts?: number` - Maximum posts to display (default: SITE.postPerIndex)

### ContentBlock.astro
- **Purpose**: Layout for content pages with breadcrumbs and title
- **Props**:
  - `pageTitle: string | [string, string]` - Page title (string or array for transitions)
  - `pageDesc?: string` - Optional page description
  - `titleTransition?: string` - Transition name for array titles

### FooterBlock.astro
- **Purpose**: Footer with social links and copyright
- **Props**:
  - `noMarginTop?: boolean` - Remove top margin (default: false)
  - `showSocials?: boolean` - Show social links (default: true)
  - `showCopyright?: boolean` - Show copyright (default: true)
  - `customContent?: string` - Custom HTML content

## Usage Example

```astro
---
import BlockLayout from "@/layouts/BlockLayout.astro";
import NavigationBlock from "@/components/blocks/NavigationBlock.astro";
import HeroBlock from "@/components/blocks/HeroBlock.astro";
import FeaturedPostsBlock from "@/components/blocks/FeaturedPostsBlock.astro";
import FooterBlock from "@/components/blocks/FooterBlock.astro";
---

<BlockLayout>
  <NavigationBlock />
  
  <main>
    <div class="mx-auto max-w-app px-4">
      <HeroBlock 
        title="My Blog"
        description="Welcome to my blog"
      />
      
      <FeaturedPostsBlock posts={featuredPosts} />
    </div>
  </main>
  
  <FooterBlock />
</BlockLayout>
```

## Benefits

1. **Modularity**: Each block is independent and reusable
2. **Maintainability**: Easy to update individual components
3. **Flexibility**: Mix and match blocks for different layouts
4. **Consistency**: Unified styling and behavior
5. **Type Safety**: Full TypeScript support with proper interfaces

## Styling

Block-specific styles are defined in `src/styles/global.css` under the `@layer components` section. Each block has its own CSS classes for easy customization.

## Customization

To customize a block:
1. Modify the component file in this directory
2. Update the corresponding CSS classes in `global.css`
3. Adjust props interface if needed
4. Test across different pages and screen sizes
