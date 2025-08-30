import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#FFFAE3',
          100: '#FFF9D6',
          900: '#2E2E2E',
          950: '#1A1A1A',
        },
        accent: {
          pink: '#FF6B6B',
          turquoise: '#4ECDC4',
          blue: '#3DB3FC',        // canlı mavi
          orange: '#FC8B3D',      // turuncu
          yellow: '#FCC43D',      // sarı
          darkGreen: '#1F4E3D',   // koyu yeşil
          green: '#2D6B4F',       // orta yeşil
        },
        background: {
          cream: '#FFFAE3',
          light: '#FFF9D6',
          darkBlue: '#608CA7',    // koyu mavi - can be background
          darkGray: '#4A4F52',    // koyu gri
          seaGreen: '#2E8B57',    // game pages background color
        },
        text: {
          primary: '#2E2E2E',
          secondary: '#4A4A4A',
        },
        custom: {
          brown1: '#A77D60',      // en sondaki renk
          brown2: '#A79260',      // en sondan bi onceki
        }
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
       },
      backgroundSize: {
        '3x3': '3px 3px',
      },
    },
  },
  plugins: [],
};

export default config;
