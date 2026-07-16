import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LocalGPT",
  description: "Private, local retrieval-augmented chat",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="bg-black">
      <body className="antialiased h-screen overflow-hidden flex flex-col">
        {children}
      </body>
    </html>
  );
}
