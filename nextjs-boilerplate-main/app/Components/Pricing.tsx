"use client";
import React, { useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

const points = [
  { text: "Unlimited access to AI-generated components", isIncluded: true },
  { text: "Advanced AI tools", isIncluded: true },
  { text: "Monthly AI training sessions", isIncluded: true },
  { text: "Priority support", isIncluded: true },
  { text: "Custom AI solutions", isIncluded: false },
  { text: "One-on-one AI consultancy", isIncluded: false },
];

const starIcon = (
  <svg
    version="1.1"
    id="Layer_1"
    xmlns="http://www.w3.org/2000/svg"
    xmlnsXlink="http://www.w3.org/1999/xlink"
    x="0px"
    y="0px"
    viewBox="0 0 122.88 122.88"
    xmlSpace="preserve"
    className="w-4 h-4 mr-2"
  >
    <g>
      <path
        className="st0"
        style={{ fillRule: 'evenodd', clipRule: 'evenodd', fill: 'currentColor' }}
        d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 
        c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 
        c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"
      />
    </g>
  </svg>
);

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, delay: i * 0.3 },
  }),
};

const pricingVariants = (direction) => ({
  hidden: { opacity: 0, x: direction === 'left' ? -100 : 100 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.7 },
  },
});

const PricingSection = () => {
  const [isMonthly, setIsMonthly] = useState(true);
  const handleToggle = () => {
    setIsMonthly(!isMonthly);
  };

  const titleRef = useRef(null);
  const subtitleRef = useRef(null);
  const switchRef = useRef(null);
  const leftBoxRef = useRef(null);
  const rightBoxRef = useRef(null);

  const isTitleInView = useInView(titleRef, { once: true, threshold: 0.3 });
  const isSubtitleInView = useInView(subtitleRef, { once: true, threshold: 0.4 });
  const isSwitchInView = useInView(switchRef, { once: true, threshold: 0.5 });
  const isLeftBoxInView = useInView(leftBoxRef, { once: true, threshold: 0.6 });
  const isRightBoxInView = useInView(rightBoxRef, { once: true, threshold: 0.6 });

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center space-y-5 sm:space-y-10 px-4 max-w-7xl mx-auto py-20">
        <motion.h1
          ref={titleRef}
          className="text-white text-4xl sm:text-5xl mb-0 sm:mb-4 tracking-tight font-semibold text-center"
          initial="hidden"
          animate={isTitleInView ? 'visible' : 'hidden'}
          custom={0}
          variants={fadeInUp}
        >
          GenAI Pricing Plans
        </motion.h1>
        <motion.p
          ref={subtitleRef}
          className="text-gray-300 text-md sm:text-xl text-center max-w-4xl"
          initial="hidden"
          animate={isSubtitleInView ? 'visible' : 'hidden'}
          custom={1}
          variants={fadeInUp}
        >
          Choose a plan that fits your needs and take advantage of our AI-powered tools for generating web components.
        </motion.p>
        <motion.div
          ref={switchRef}
          className="w-full flex justify-center"
          initial="hidden"
          animate={isSwitchInView ? 'visible' : 'hidden'}
          custom={2}
          variants={fadeInUp}
        >
          <div className='bg-[#273040] rounded-lg p-1 mb-16'>
            <div className="relative flex items-center bg-[#273040] rounded-lg p-1 ">
              <motion.div
                className="absolute top-0 left-0 w-1/2 h-full bg-[#141d26] rounded-lg"
                animate={{ x: isMonthly ? 0 : '100%' }}
                transition={{ type: 'spring', stiffness: 150 }}
              />
              <div
                className={`relative py-1 px-4 cursor-pointer rounded-lg pr-8 ${isMonthly ? 'text-white font-semibold' : 'text-gray-400'}`}
                onClick={handleToggle}
              >
                <span>Monthly Payment</span>
              </div>
              <div
                className={`relative py-1 px-4 cursor-pointer rounded-lg ${!isMonthly ? 'text-white font-semibold' : 'text-gray-400'}`}
                onClick={handleToggle}
              >
                <span>Annual Payment</span>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="flex flex-col md:flex-row space-y-10 md:space-y-0 md:space-x-10">
          <motion.div
            ref={leftBoxRef}
            className="relative bg-[#141d26] border-2 border-[#232c39] max-w-[450px] h-full flex flex-col items-left rounded-xl p-6"
            initial="hidden"
            animate={isLeftBoxInView ? 'visible' : 'hidden'}
            variants={pricingVariants('left')}
          >
            <div className="bg-[#2b2836] border-2 border-[#232c39] rounded-2xl flex items-center justify-center w-12 h-12 p-3">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                className="w-6 h-6 text-white"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
              </svg>
            </div>
            <h2 className="text-white text-3xl mb-4 mt-4">Basic Plan</h2>
            <p className="text-gray-400 mr-6">For individuals looking to explore AI-generated web components and tools.</p>
            <div className="flex items-baseline mt-8">
              <motion.h2 
                className="text-white text-6xl mb-4"
                key={isMonthly ? "monthly" : "annual"}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                {isMonthly ? "£80" : "£800"}
              </motion.h2>
              <span className="text-gray-400 text-sm mb-4 ml-2 self-end">/ per {isMonthly ? "month" : "year"}</span>
            </div>
            <p className="text-gray-400 mr-6 mt-4">What&apos;s Included?</p>
            <ul className="mt-4">
              {points.map((point, index) => (
                <li key={index} className="flex items-center mb-4">
                  <div className={`w-4 h-4 mr-2 ${point.isIncluded ? 'text-white' : 'text-gray-400'}`}>
                    {starIcon}
                  </div>
                  <span className={`${point.isIncluded ? 'text-white' : 'text-gray-400'}`}>{point.text}</span>
                </li>
              ))}
            </ul>
            <motion.button 
              whileHover={{ 
                backgroundColor: '#00DCDC',
                color: '#141d26',
                boxShadow: '0px 0px 12px rgba(0, 220, 220, 0.6)',
                transition: { duration: 0.3 }
              }}
              className="mt-6 bg-white text-[#141d26] font-semibold py-3 px-4 rounded-xl w-full text-center"
            >
              Buy Now
            </motion.button>
          </motion.div>

          <motion.div
            ref={rightBoxRef}
            className="relative bg-[#141d26] border-2 border-[#00DCDC] max-w-[450px] h-full flex flex-col items-left rounded-xl p-6"
            initial="hidden"
            animate={isRightBoxInView ? 'visible' : 'hidden'}
            variants={pricingVariants('right')}
          >
            <div className="bg-[#2b2836] border-2 border-[#232c39] rounded-2xl flex items-center justify-center w-12 h-12 p-3">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                className="w-6 h-6 text-white"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h2 className="text-white text-3xl mb-4 mt-4">Pro Plan</h2>
            <p className="text-gray-400 mr-6">Ideal for professionals who need advanced AI tools and personalized support.</p>
            <div className="flex items-baseline mt-8">
              <motion.h2 
                className="text-white text-6xl mb-4"
                key={isMonthly ? "monthly2" : "annual2"}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                {isMonthly ? "£300" : "£3000"}
              </motion.h2>
              <span className="text-gray-400 text-sm mb-4 ml-2 self-end">/ per {isMonthly ? "month" : "year"}</span>
            </div>
            <p className="text-gray-400 mr-6 mt-4">What&apos;s Included?</p>
            <ul className="mt-4">
              {points.map((point, index) => (
                <li key={index} className="flex items-center mb-4">
                  <div className={`w-4 h-4 mr-2 text-white`}>
                    {starIcon}
                  </div>
                  <span className="text-white">{point.text}</span>
                </li>
              ))}
            </ul>
            <motion.button 
              whileHover={{ 
                backgroundColor: '#00DCDC',
                color: '#141d26',
                boxShadow: '0px 0px 12px rgba(0, 220, 220, 0.6)',
                transition: { duration: 0.3 }
              }}
              className="mt-6 bg-white text-[#141d26] font-semibold py-3 px-4 rounded-xl w-full text-center"
            >
              Buy Now
            </motion.button>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default PricingSection;
