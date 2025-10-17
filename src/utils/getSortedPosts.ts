import type { CollectionEntry } from "astro:content";
import postFilter from "./postFilter";

const getSortedPosts = (posts: CollectionEntry<"blog">[]) => {
  return posts
    .filter(postFilter)
    .sort(
      (a, b) => {
        const dateA = a.data.pubDatetime ? new Date(a.data.pubDatetime).getTime() / 1000 : 0;
        const dateB = b.data.pubDatetime ? new Date(b.data.pubDatetime).getTime() / 1000 : 0;
        return Math.floor(dateB) - Math.floor(dateA);
      }
    );
};

export default getSortedPosts;
