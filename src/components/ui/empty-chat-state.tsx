"use client";

import { useEffect, useRef, useCallback } from "react";
import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
    ArrowUpIcon,
    Paperclip,
    PlusIcon,
    X,
    FileText,
} from "lucide-react";
import { AttachedFile } from "@/lib/types";

interface UseAutoResizeTextareaProps {
    minHeight: number;
    maxHeight?: number;
}

function useAutoResizeTextarea({
    minHeight,
    maxHeight,
}: UseAutoResizeTextareaProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const adjustHeight = useCallback(
        (reset?: boolean) => {
            const textarea = textareaRef.current;
            if (!textarea) return;

            if (reset) {
                textarea.style.height = `${minHeight}px`;
                return;
            }

            // Temporarily shrink to get the right scrollHeight
            textarea.style.height = `${minHeight}px`;

            // Calculate new height
            const newHeight = Math.max(
                minHeight,
                Math.min(
                    textarea.scrollHeight,
                    maxHeight ?? Number.POSITIVE_INFINITY
                )
            );

            textarea.style.height = `${newHeight}px`;
        },
        [minHeight, maxHeight]
    );

    useEffect(() => {
        // Set initial height
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = `${minHeight}px`;
        }
    }, [minHeight]);

    // Adjust height on window resize
    useEffect(() => {
        const handleResize = () => adjustHeight();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, [adjustHeight]);

    return { textareaRef, adjustHeight };
}

interface EmptyChatStateProps {
    onSendMessage: (message: string, attachedFiles?: AttachedFile[]) => void;
    disabled?: boolean;
    placeholder?: string;
}

export function EmptyChatState({ 
    onSendMessage, 
    disabled = false, 
    placeholder = "Ask localgpt a question..." 
}: EmptyChatStateProps) {
    const [value, setValue] = useState("");
    const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { textareaRef, adjustHeight } = useAutoResizeTextarea({
        minHeight: 60,
        maxHeight: 200,
    });

    const handleSend = () => {
        if ((value.trim() || attachedFiles.length > 0) && !disabled) {
            onSendMessage(value.trim(), attachedFiles);
            setValue("");
            setAttachedFiles([]);
            adjustHeight(true);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleFileAttach = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files) return;

        const newFiles: AttachedFile[] = [];
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (file.type === 'application/pdf' || 
                file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
                file.type === 'application/msword' ||
                file.type === 'text/html' ||
                file.type === 'text/markdown' ||
                file.type === 'text/plain' ||
                file.name.toLowerCase().endsWith('.pdf') ||
                file.name.toLowerCase().endsWith('.docx') ||
                file.name.toLowerCase().endsWith('.doc') ||
                file.name.toLowerCase().endsWith('.html') ||
                file.name.toLowerCase().endsWith('.htm') ||
                file.name.toLowerCase().endsWith('.md') ||
                file.name.toLowerCase().endsWith('.txt')) {
                newFiles.push({
                    id: crypto.randomUUID(),
                    name: file.name,
                    size: file.size,
                    type: file.type,
                    file: file,
                });
            }
        }

        setAttachedFiles(prev => [...prev, ...newFiles]);
        
        // Reset the input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }

        // --- NEW: Immediately trigger upload when files are selected ---
        if (newFiles.length > 0) {
            onSendMessage("", newFiles);
            // Clear the local attachment state as the parent now handles it
            setAttachedFiles([]); 
        }
    };

    const removeFile = (fileId: string) => {
        setAttachedFiles(prev => prev.filter(f => f.id !== fileId));
    };

    const formatFileSize = (bytes: number) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div className="flex flex-col items-center justify-center h-full w-full max-w-4xl mx-auto p-4 space-y-8">
            <h1 className="text-4xl font-bold text-white">
                What can I help you find?
            </h1>

            <div className="w-full">
                {/* Attached Files Display */}
                {attachedFiles.length > 0 && (
                    <div className="mb-4 space-y-2">
                        <div className="text-sm text-gray-400 font-medium">Attached Files:</div>
                        <div className="space-y-2">
                            {attachedFiles.map((file) => (
                                <div key={file.id} className="flex items-center gap-3 bg-gray-800 rounded-lg p-3">
                                    <FileText className="w-5 h-5 text-red-400" />
                                    <div className="flex-1 min-w-0">
                                        <div className="text-sm text-white truncate">{file.name}</div>
                                        <div className="text-xs text-gray-400">{formatFileSize(file.size)}</div>
                                    </div>
                                    {/* The remove button is commented out as the parent will manage the state now */}
                                    {/* <button
                                        onClick={() => removeFile(file.id)}
                                        className="p-1 hover:bg-gray-700 rounded transition-colors"
                                    >
                                        <X className="w-4 h-4 text-gray-400 hover:text-white" />
                                    </button> */}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div className="relative bg-neutral-900 rounded-xl border border-neutral-800">
                    <div className="overflow-y-auto">
                        <Textarea
                            ref={textareaRef}
                            value={value}
                            onChange={(e) => {
                                setValue(e.target.value);
                                adjustHeight();
                            }}
                            onKeyDown={handleKeyDown}
                            placeholder={attachedFiles.length > 0 ? "Ask questions about your attached files..." : placeholder}
                            disabled={disabled}
                            className={cn(
                                "w-full px-4 py-3",
                                "resize-none",
                                "bg-transparent",
                                "border-none",
                                "text-white text-sm",
                                "focus:outline-none",
                                "focus-visible:ring-0 focus-visible:ring-offset-0",
                                "placeholder:text-neutral-500 placeholder:text-sm",
                                "min-h-[60px]",
                                disabled && "opacity-50 cursor-not-allowed"
                            )}
                            style={{
                                overflow: "hidden",
                            }}
                        />
                    </div>

                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,.docx,.doc,.html,.htm,.md,.txt"
                        multiple
                        onChange={handleFileChange}
                        className="hidden"
                    />

                    <div className="flex items-center justify-between p-3">
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                onClick={handleFileAttach}
                                disabled={disabled}
                                className="group p-2 hover:bg-neutral-800 rounded-lg transition-colors flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Attach PDF files"
                            >
                                <Paperclip className="w-4 h-4 text-white" />
                                <span className="text-xs text-zinc-400 hidden group-hover:inline transition-opacity">
                                    Attach PDF
                                </span>
                            </button>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                disabled={disabled}
                                className="px-2 py-1 rounded-lg text-sm text-zinc-400 transition-colors border border-dashed border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800 flex items-center justify-between gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <PlusIcon className="w-4 h-4" />
                                Project
                            </button>
                            <button
                                type="button"
                                onClick={handleSend}
                                disabled={disabled || (!value.trim() && attachedFiles.length === 0)}
                                className={cn(
                                    "px-1.5 py-1.5 rounded-lg text-sm transition-colors border border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800 flex items-center justify-between gap-1",
                                    (value.trim() || attachedFiles.length > 0) && !disabled
                                        ? "bg-white text-black hover:bg-gray-200"
                                        : "text-zinc-400",
                                    "disabled:opacity-50 disabled:cursor-not-allowed"
                                )}
                            >
                                <ArrowUpIcon
                                    className={cn(
                                        "w-4 h-4",
                                        (value.trim() || attachedFiles.length > 0) && !disabled
                                            ? "text-black"
                                            : "text-zinc-400"
                                    )}
                                />
                                <span className="sr-only">Send</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}    