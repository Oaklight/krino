"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const stored = localStorage.getItem("theme") as Theme | null;
    if (stored) setTheme(stored);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") {
      localStorage.removeItem("theme");
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.classList.toggle("dark", prefersDark);
    } else {
      localStorage.setItem("theme", theme);
      root.classList.toggle("dark", theme === "dark");
    }
  }, [theme]);

  const cycle = () => {
    setTheme((t) => (t === "light" ? "dark" : t === "dark" ? "system" : "light"));
  };

  const icon = theme === "dark" ? "\u{1F319}" : theme === "light" ? "☀️" : "\u{1F4BB}";

  return (
    <button
      onClick={cycle}
      className="p-2 rounded-[var(--radius)] hover:bg-bg-hover transition-colors text-sm"
      aria-label={`Theme: ${theme}`}
      title={`Theme: ${theme}`}
    >
      {icon}
    </button>
  );
}
