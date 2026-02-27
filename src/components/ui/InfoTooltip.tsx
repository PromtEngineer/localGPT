import { useState } from "react";
import { Info } from "lucide-react";

interface Props {
  text: string;
  className?: string;
  size?: number;
}

// A lightweight hover / focus tooltip used next to form labels.
// It shows a small Info icon; on hover (or focus) a dark glassy popover appears.
export function InfoTooltip({ text, className = "", size = 14 }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className={`relative inline-block align-middle ${className}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      tabIndex={0}
    >
      <Info size={size} className="text-gray-400 hover:text-white cursor-pointer" />
      {open && (
        <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-56 bg-black/80 backdrop-blur-sm text-gray-200 text-xs px-3 py-2 rounded shadow-lg z-50 normal-case whitespace-normal break-words">
          {text}
        </div>
      )}
    </span>
  );
} 