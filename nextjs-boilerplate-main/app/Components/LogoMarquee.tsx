// app/Components/LogoMarquee.tsx
"use client"
import { motion } from 'framer-motion';
import Image from 'next/image';

const logos = [
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/10_jsx7lx.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/18_gzkxpm.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/17_nvkboi.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/10_jsx7lx.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/18_gzkxpm.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/17_nvkboi.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/10_jsx7lx.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/18_gzkxpm.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/17_nvkboi.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/10_jsx7lx.png',
  'https://res.cloudinary.com/dl2adjye7/image/upload/v1712483444/18_gzkxpm.png',
];

const LogoMarquee = () => {
  return (
    <div className="relative max-w-[1800px] mx-auto overflow-hidden bg-transparent">
      <motion.div
        className="flex space-x-8"
        initial={{ x: 0 }}
        animate={{ x: '-100%' }}
        transition={{ repeat: Infinity, duration: 30, ease: 'linear' }}
      >
        {[...logos, ...logos].map((logo, index) => (
          <Image key={index} src={logo} alt={`logo-${index}`} width={200} height={200} className="brightness-0 invert" />
        ))}
      </motion.div>
      <div className="absolute inset-y-0 left-0 w-32 bg-gradient-to-r from-slate-900 to-transparent pointer-events-none"></div>
      <div className="absolute inset-y-0 right-0 w-32 bg-gradient-to-l from-slate-900 to-transparent pointer-events-none"></div>
    </div>
  );
};

export default LogoMarquee;
