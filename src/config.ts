export const SITE = {
  website: "https://astro-paper.pages.dev/", // replace this with your deployed domain
  author: "Tấn Đạt",
  profile: "https://satnaing.dev/",
  desc: "Portfolio AI & Machine Learning của Tấn Đạt, kèm blog học tập theo module.",
  title: "Tấn Đạt Portfolio",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: false,
  postPerIndex: 6,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: false,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Edit page",
    url: "https://github.com/tandat8896/tandat8896.github.io/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "vi", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Bangkok", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
