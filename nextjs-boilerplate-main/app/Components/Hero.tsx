"use client";
import { useState } from 'react';
import { motion } from "framer-motion";
import { GetStarted } from './Buttons/GetStarted';
import CustomCursor from './CustomCursor';
import Pricing from './Buttons/Pricing';


export default function Hero() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [cursorVisible, setCursorVisible] = useState(false);

  const handleMouseEnter = () => {
    setCursorVisible(true);
  };

  const handleMouseLeave = () => {
    setCursorVisible(false);
  };

  const draw = {
    hidden: { pathLength: 0, opacity: 0 },
    visible: {
      pathLength: 1,
      opacity: 1,
      transition: {
        pathLength: { delay: 0.5, type: "spring", duration: 2, bounce: 0 },
        opacity: { delay: 0.5, duration: 1 }
      }
    }
  };

  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, delay: i * 0.3 }
    })
  };

  const handlePricingClick = () => {
    const element = document.getElementById('pricingSection');
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  return (
    <div className="relative isolate">
      <div className="py-28 sm:py-40 lg:pb-64">
        <div className="mx-auto max-w-6xl px-6 lg:px-8">
          <div className="mx-auto max-w-4xl text-center">
            <motion.h1
              className="text-4xl font-bold tracking-tight text-white sm:text-6xl font-alliance"
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              custom={0}
            >
             <span className="relative whitespace-nowrap text-teal-700 font-alliance">
          <motion.svg
            aria-hidden="true"
            viewBox="0 0 418 42"
            className="absolute left-0 top-1/3 fill-teal-400"
            preserveAspectRatio="xMidYMid meet"
            width="100%"
            height="100%"
            initial="hidden"
            animate="visible"
          >
            <motion.path
              d="M203.371.916c-26.013-2.078-76.686 1.963-124.73 9.946L67.3 12.749C35.421 18.062 18.2 21.766 6.004 25.934 1.244 27.561.828 27.778.874 28.61c.07 1.214.828 1.121 9.595-1.176 9.072-2.377 17.15-3.92 39.246-7.496C123.565 7.986 157.869 4.492 195.942 5.046c7.461.108 19.25 1.696 19.17 2.582-.107 1.183-7.874 4.31-25.75 10.366-21.992 7.45-35.43 12.534-36.701 13.884-2.173 2.308-.202 4.407 4.442 4.734 2.654.187 3.263.157 15.593-.78 35.401-2.686 57.944-3.488 88.365-3.143 46.327.526 75.721 2.23 130.788 7.584 19.787 1.924 20.814 1.98 24.557 1.332l.066-.011c1.201-.203"
              variants={draw}
              strokeWidth="4"
              fill="none"
              stroke="#0D9488"
            />
          </motion.svg>
          <span className="relative text-teal-400 font-alliance">Transform</span>
        </span>
              {' '}Your UI, AI-Generated Components
            </motion.h1>
            
            <motion.p
              className="mt-8 text-base leading-7 text-white"
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              custom={1}
            >
             Elevate your web development with our AI-powered React components. Effortlessly create stunning, responsive interfaces in minutes, saving you time and enhancing the user experience with cutting-edge technology.
            </motion.p>
            <motion.div
              className="mt-8 flex items-center justify-center gap-x-4"
              initial="hidden"
              animate="visible"
              variants={fadeInUp}
              custom={2}
            >
              <GetStarted name={"Get Started"} />
              <Pricing name={"Pricing"} onClick={handlePricingClick} />
            </motion.div>
          </div>
          <div className="mt-12 flow-root sm:mt-20 relative">
            <motion.div 
              className="-m-2 rounded-lg bg-slate-800 p-2 ring-1 ring-inset ring-gray-900/10 lg:-m-4 lg:rounded-xl lg:p-4 cursor-none"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
            >
              <img
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                src="https://res.cloudinary.com/dl2adjye7/image/upload/v1714481948/hero_x6wubt.png"
                alt="App screenshot"
                className="rounded shadow-xl ring-1 ring-gray-900/10 w-full"
              />
            </motion.div>
          </div>
          <CustomCursor isVisible={cursorVisible} text={"You"} />
        </div>
      </div>
    </div>
  )
}
