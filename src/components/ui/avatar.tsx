"use client"

import * as React from "react"
import * as AvatarPrimitive from "@radix-ui/react-avatar"

import { cn } from "@/lib/utils"

/**
 * Avatar root component that provides the container for avatar content.
 * Renders a circular container with default styling that can be customized.
 * 
 * @param className - Additional CSS classes to apply to the avatar container
 * @param props - All other props are forwarded to the underlying Radix Avatar Root component
 * @returns JSX element representing the avatar container
 */
function Avatar({
  className,
  ...props
}: React.ComponentProps<typeof AvatarPrimitive.Root>) {
  return (
    <AvatarPrimitive.Root
      data-slot="avatar"
      className={cn(
        "relative flex size-8 shrink-0 overflow-hidden rounded-full",
        className
      )}
      {...props}
    />
  )
}

/**
 * Avatar image component for displaying the main avatar image.
 * Automatically handles image loading states and fallback behavior.
 * 
 * @param className - Additional CSS classes to apply to the image element
 * @param props - All other props are forwarded to the underlying Radix Avatar Image component
 * @returns JSX element representing the avatar image
 */
function AvatarImage({
  className,
  ...props
}: React.ComponentProps<typeof AvatarPrimitive.Image>) {
  return (
    <AvatarPrimitive.Image
      data-slot="avatar-image"
      className={cn("aspect-square size-full", className)}
      {...props}
    />
  )
}

/**
 * Avatar fallback component that displays when the main image fails to load.
 * Typically contains initials, icons, or other placeholder content.
 * 
 * @param className - Additional CSS classes to apply to the fallback element
 * @param props - All other props are forwarded to the underlying Radix Avatar Fallback component
 * @returns JSX element representing the avatar fallback content
 */
function AvatarFallback({
  className,
  ...props
}: React.ComponentProps<typeof AvatarPrimitive.Fallback>) {
  return (
    <AvatarPrimitive.Fallback
      data-slot="avatar-fallback"
      className={cn(
        "bg-muted flex size-full items-center justify-center rounded-full text-black",
        className
      )}
      {...props}
    />
  )
}

export { Avatar, AvatarImage, AvatarFallback }
