import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google"; // Import the Uniform font
import "./globals.css";
import Navbar from "./Components/Navbar";
import { UserProvider } from "./contexts/UserContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Cards Against Humanity",
  description: "AI-powered humor generation with personalized learning and feedback",
  openGraph: {
    title: "Cards Against Humanity",
    description: "AI-powered humor generation with personalized learning and feedback",
    siteName: "AI CAH",
    images: [
      {
        url: "https://res.cloudinary.com/dl2adjye7/image/upload/v1716922972/DynaUI_rfzbgc.png",
        width: 800,
        height: 600,
        alt: "AI CAH Share Image",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Cards Against Humanity",
    description: "AI-powered humor generation with personalized learning and feedback",
    images:
      "https://res.cloudinary.com/dl2adjye7/image/upload/v1716922972/DynaUI_rfzbgc.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.png" />
      </head>
      <body className={inter.className}>
        {/* Apply the fonts */}
        <UserProvider>
          <Navbar />
          {children}
        </UserProvider>
      </body>
    </html>
  );
}
