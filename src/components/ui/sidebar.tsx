"use client";

import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { motion } from "framer-motion";
import {
  ChevronsUpDown,
  LogOut,
  MessagesSquare,
  Plus,
  Settings,
  UserCircle,
} from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";

const sidebarVariants = {
  open: {
    width: "15rem",
  },
  closed: {
    width: "3.05rem",
  },
};

const contentVariants = {
  open: { display: "block", opacity: 1 },
  closed: { display: "block", opacity: 1 },
};

const variants = {
  open: {
    x: 0,
    opacity: 1,
    transition: {
      x: { stiffness: 1000, velocity: -100 },
    },
  },
  closed: {
    x: -20,
    opacity: 0,
    transition: {
      x: { stiffness: 100 },
    },
  },
};

const transitionProps = {
  type: "tween",
  ease: "easeOut",
  duration: 0.2,
  staggerChildren: 0.1,
};

const staggerVariants = {
  open: {
    transition: { staggerChildren: 0.03, delayChildren: 0.02 },
  },
};

// Mock chat sessions data
const chatSessions = [
  { id: 1, title: "React Component Help", lastMessage: "How to create a sidebar?", timestamp: "2 min ago", isActive: true },
  { id: 2, title: "TypeScript Questions", lastMessage: "Interface vs Type", timestamp: "1 hour ago", isActive: false },
  { id: 3, title: "Next.js Setup", lastMessage: "Setting up shadcn/ui", timestamp: "3 hours ago", isActive: false },
  { id: 4, title: "Tailwind CSS", lastMessage: "Dark mode implementation", timestamp: "1 day ago", isActive: false },
  { id: 5, title: "Database Design", lastMessage: "Schema optimization", timestamp: "2 days ago", isActive: false },
];

export function SessionNavBar() {
  const [isCollapsed, setIsCollapsed] = useState(true);
  
  return (
    <motion.div
      className={cn(
        "sidebar fixed left-0 z-40 h-full shrink-0 border-r border-neutral-800",
      )}
      initial={isCollapsed ? "closed" : "open"}
      animate={isCollapsed ? "closed" : "open"}
      variants={sidebarVariants}
      transition={transitionProps}
      onMouseEnter={() => setIsCollapsed(false)}
      onMouseLeave={() => setIsCollapsed(true)}
    >
      <motion.div
        className={`relative z-40 flex text-muted-foreground h-full shrink-0 flex-col bg-black transition-all`}
        variants={contentVariants}
      >
        <motion.ul variants={staggerVariants} className="flex h-full flex-col">
          <div className="flex grow flex-col items-center">
            {/* Header */}
            <div className="flex h-[54px] w-full shrink-0 border-b border-neutral-800 p-2">
              <div className="mt-[1.5px] flex w-full">
                <DropdownMenu modal={false}>
                  <DropdownMenuTrigger className="w-full" asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex w-fit items-center gap-2 px-2 text-white hover:bg-neutral-800" 
                    >
                      <Avatar className='rounded size-4'>
                        <AvatarFallback className="bg-blue-600 text-white">L</AvatarFallback>
                      </Avatar>
                      <motion.li
                        variants={variants}
                        className="flex w-fit items-center gap-2"
                      >
                        {!isCollapsed && (
                          <>
                            <p className="text-sm font-medium text-white">
                              localGPT
                            </p>
                            <ChevronsUpDown className="h-4 w-4 text-neutral-400" />
                          </>
                        )}
                      </motion.li>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="bg-neutral-900 border-neutral-800">
                    <DropdownMenuItem className="flex items-center gap-2 text-white hover:bg-neutral-800">
                      <Settings className="h-4 w-4" /> Preferences
                    </DropdownMenuItem>
                    <DropdownMenuItem className="flex items-center gap-2 text-white hover:bg-neutral-800">
                      <Plus className="h-4 w-4" /> New Chat
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            {/* Chat Sessions */}
            <div className="flex h-full w-full flex-col">
              <div className="flex grow flex-col gap-4">
                <ScrollArea className="h-16 grow p-2">
                  <div className={cn("flex w-full flex-col gap-1")}>
                    {/* New Chat Button */}
                    <Button
                      variant="ghost"
                      className="flex h-8 w-full flex-row items-center justify-start rounded-md px-2 py-1.5 text-white hover:bg-neutral-800 mb-2"
                    >
                      <Plus className="h-4 w-4" />
                      <motion.span variants={variants} className="ml-2">
                        {!isCollapsed && (
                          <p className="text-sm font-medium">New Chat</p>
                        )}
                      </motion.span>
                    </Button>
                    
                    <Separator className="w-full bg-neutral-800" />
                    
                    {/* Chat Sessions List */}
                    {chatSessions.map((session) => (
                      <div
                        key={session.id}
                        className={cn(
                          "flex h-auto w-full flex-col rounded-md px-2 py-2 transition hover:bg-neutral-800 cursor-pointer",
                          session.isActive && "bg-neutral-800"
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <MessagesSquare className="h-4 w-4 text-neutral-400 shrink-0" />
                          <motion.div variants={variants} className="flex-1 min-w-0">
                            {!isCollapsed && (
                              <div className="flex flex-col gap-1">
                                <p className="text-sm font-medium text-white truncate">
                                  {session.title}
                                </p>
                                <p className="text-xs text-neutral-400 truncate">
                                  {session.lastMessage}
                                </p>
                                <p className="text-xs text-neutral-500">
                                  {session.timestamp}
                                </p>
                              </div>
                            )}
                          </motion.div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
              
              {/* Footer */}
              <div className="flex flex-col p-2 border-t border-neutral-800">
                <Button
                  variant="ghost"
                  className="mt-auto flex h-8 w-full flex-row items-center rounded-md px-2 py-1.5 text-white hover:bg-neutral-800"
                >
                  <Settings className="h-4 w-4 shrink-0" />
                  <motion.span variants={variants}>
                    {!isCollapsed && (
                      <p className="ml-2 text-sm font-medium">Settings</p>
                    )}
                  </motion.span>
                </Button>
                
                <DropdownMenu modal={false}>
                  <DropdownMenuTrigger className="w-full">
                    <div className="flex h-8 w-full flex-row items-center gap-2 rounded-md px-2 py-1.5 transition hover:bg-neutral-800">
                      <Avatar className="size-4">
                        <AvatarFallback className="bg-blue-600 text-white text-xs">
                          U
                        </AvatarFallback>
                      </Avatar>
                      <motion.div
                        variants={variants}
                        className="flex w-full items-center gap-2"
                      >
                        {!isCollapsed && (
                          <>
                            <p className="text-sm font-medium text-white">User</p>
                            <ChevronsUpDown className="ml-auto h-4 w-4 text-neutral-400" />
                          </>
                        )}
                      </motion.div>
                    </div>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent sideOffset={5} className="bg-neutral-900 border-neutral-800">
                    <div className="flex flex-row items-center gap-2 p-2">
                      <Avatar className="size-6">
                        <AvatarFallback className="bg-blue-600 text-white">
                          U
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex flex-col text-left">
                        <span className="text-sm font-medium text-white">
                          User
                        </span>
                        <span className="line-clamp-1 text-xs text-neutral-400">
                          user@example.com
                        </span>
                      </div>
                    </div>
                    <DropdownMenuSeparator className="bg-neutral-800" />
                    <DropdownMenuItem className="flex items-center gap-2 text-white hover:bg-neutral-800">
                      <UserCircle className="h-4 w-4" /> Profile
                    </DropdownMenuItem>
                    <DropdownMenuItem className="flex items-center gap-2 text-white hover:bg-neutral-800">
                      <LogOut className="h-4 w-4" /> Sign out
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </div>
        </motion.ul>
      </motion.div>
    </motion.div>
  );
} 