// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

export const site = {
  title: "Xinsong Feng | 冯欣淞 - Homepage", // required
  favicon: "/favicon.png", // required
  description: "MSc in ECE",
  author: "Xinsong Feng", // required
  author_CN: "冯欣淞", // required
  avatar: "/xsfeng.jpg", // required
  toc: {
    enable: true, // Display the table of contents on the right side of the post
    depth: 2, // Maximum heading depth to show in the table, from 1 to 3
  },
};

export const blog = {
  postsPerPage: 7, // Posts to display per page
};

export const info = {
  location: {
    icon: "fas fa-map-marker-alt",
    color: "text-blue-500",
    name: "Los Angeles",
  },
  social: [
    {
      icon: "fas fa-envelope",
      color: "text-green-500",
      name: "Email",
      link: "mailto:xsfeng@g.ucla.com",
    },
    {
      icon: "fas fa-graduation-cap",
      color: "text-red-500",
      name: "Google Scholar",
      link: "https://scholar.google.com/citations?hl=en&user=664D7CoAAAAJ",
    },
    {
      icon: "fab fa-github",
      color: "text-gray-700",
      name: "GitHub",
      link: "https://github.com/xfeng01",
    },
    {
      icon: "fas fa-file-contract",
      color: "text-blue-500",
      name: "Resume",
      link: "/Xinsong_Feng_CV.pdf",
    },
  ],
};
