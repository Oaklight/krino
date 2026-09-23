import fs from "fs";
import path from "path";
import type { DashboardData } from "./types";

export function loadDashboardData(): DashboardData {
  const filePath = path.join(process.cwd(), "public", "results", "eval_results.json");
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as DashboardData;
}
