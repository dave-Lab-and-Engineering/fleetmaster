window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex",
  },
};

// Re-run MathJax after every client-side navigation/reload by Material
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    // Clear queued typesetting (if any) and re-typeset
    MathJax.typesetClear && MathJax.typesetClear();
    MathJax.typesetPromise();
  }
});
