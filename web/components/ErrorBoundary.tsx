"use client";

import React from "react";

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="border border-red rounded-[var(--radius)] p-6 text-center">
            <p className="font-medium text-red mb-2">Something went wrong</p>
            <p className="text-sm text-text-muted">{this.state.error?.message}</p>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
