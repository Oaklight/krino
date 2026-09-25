export default function Footer() {
  return (
    <footer className="border-t border-border mt-auto py-6">
      <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-text-muted">
        <span>krino &mdash; typed decision models for calibrated probabilities</span>
        <div className="flex gap-4">
          <a
            href="https://github.com/Oaklight/krino"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-text transition-colors"
          >
            GitHub
          </a>
          <a
            href="https://docs.typesafe.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-text transition-colors"
          >
            TypeSafe Docs
          </a>
        </div>
      </div>
    </footer>
  );
}
