"use client"
import React, { useState, useRef, useEffect } from 'react';
import { FaArrowCircleRight } from 'react-icons/fa';
import { motion, useInView } from 'framer-motion';
import CustomCursor from './CustomCursor';

const BentoBox = () => {
  const [cursorText, setCursorText] = useState('');
  const [isCursorVisible, setIsCursorVisible] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const ref1 = useRef(null);
  const ref2 = useRef(null);
  const ref3 = useRef(null);
  const videoRef1 = useRef(null);
  const videoRef2 = useRef(null);
  const videoRef3 = useRef(null);

  const isInView1 = useInView(ref1, { once: true });
  const isInView2 = useInView(ref2, { once: true });
  const isInView3 = useInView(ref3, { once: true });

  const sectionRef = useRef(null);
  const isSectionInView = useInView(sectionRef, { once: true });

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const handleMouseEnter = (text: string, videoRef: React.RefObject<HTMLVideoElement>) => {
    if (!isMobile) {
      setCursorText(text);
      setIsCursorVisible(true);
      if (videoRef && videoRef.current) {
        videoRef.current.play();
      }
    }
  };

  const handleMouseLeave = (videoRef: React.RefObject<HTMLVideoElement>) => {
    if (!isMobile) {
      setIsCursorVisible(false);
      if (videoRef && videoRef.current) {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
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

  const draw = {
    hidden: { pathLength: 0, opacity: 0 },
    visible: {
      pathLength: 1,
      opacity: 1,
      transition: {
        pathLength: { delay: 0.3, type: 'spring', duration: 1.5, bounce: 0 },
        opacity: { delay: 0.3, duration: 0.8 }
      }
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-4 pt-40 sm:pt-72">
      <div ref={sectionRef} className="mx-auto max-w-4xl text-center">
        <motion.h1
          className="text-3xl font-bold tracking-tight text-white sm:text-5xl text-center"
          initial="hidden"
          animate={isSectionInView ? 'visible' : 'hidden'}
          variants={fadeInUp}
          custom={0}
        >
          <span className="relative whitespace-nowrap text-teal-800 font-alliance">
            <motion.svg
              aria-hidden="true"
              viewBox="0 0 418 42"
              className="absolute left-0 top-1/3 fill-teal-500"
              preserveAspectRatio="xMidYMid meet"
              width="100%"
              height="100%"
              initial="hidden"
              animate="visible"
            >
              <motion.path
                d="M203.371.916c-26.013-2.078-76.686 1.963-124.73 9.946L67.3 12.749C35.421 18.062 18.2 21.766 6.004 25.934 1.244 27.561.828 27.778.874 28.61c.07 1.214.828 1.121 9.595-1.176 9.072-2.377 17.15-3.92 39.246-7.496C123.565 7.986 157.869 4.492 195.942 5.046c7.461.108 19.25 1.696 19.17 2.582-.107 1.183-7.874 4.31-25.75 10.366-21.992 7.45-35.43 12.534-36.701 13.884-2.173 2.308-.202 4.407 4.442 4.734 2.654.187 3.263.157 15.593-.78 35.401-2.686 57.944-3.488 88.365-3.143 46.327.526 75.721 2.23 130.788 7.584 19.787 1.924 20.814 1.98 24.557 1.332l.066-.011c1.201-.203"
                variants={draw}
                strokeWidth="6"
                fill="none"
                stroke="#009080"
              />
            </motion.svg>
            <span className="relative text-teal-500 font-alliance">GenUI</span>
          </span>
          {' '}Innovative AI-Generated Web Components
        </motion.h1>
        <motion.p
          className="mt-10 text-lg leading-8 text-gray-200 max-w-2xl mx-auto"
          initial="hidden"
          animate={isSectionInView ? 'visible' : 'hidden'}
          variants={fadeInUp}
          custom={1}
        >
          Discover the future of web design with AI-generated components that enhance your workflow and boost productivity, all in one seamless platform.
        </motion.p>
      </div>
      <CustomCursor isVisible={isCursorVisible} text={cursorText} />
      <motion.div
        ref={ref1}
        className="bg-[#121416] border-2 border-[#28292b] h-auto mb-4 rounded-3xl shadow-lg flex flex-col lg:flex-row p-3 hover:shadow-[0_0_15px_5px_rgba(0,255,255,0.3)] transition-shadow duration-300 sm:cursor-none mt-20"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: isInView1 ? 1 : 0, y: isInView1 ? 0 : -50 }}
        transition={{ duration: 0.8 }}
        onMouseEnter={() => handleMouseEnter('You', videoRef1)}
        onMouseLeave={() => handleMouseLeave(videoRef1)}
      >
        <div className="w-full lg:w-1/2 flex flex-col justify-between p-5">
          <motion.div
            initial="hidden"
            animate={isInView1 ? 'visible' : 'hidden'}
            variants={fadeInUp}
            custom={0}
          >
            <h1 className="text-2xl font-semibold text-gray-100 mb-2 lg:mr-14">AI-Generated Components for Fast Web Development</h1>
            <p className="text-gray-400 lg:mr-14 mt-4">Accelerate your web projects with ready-to-use, AI-generated components that seamlessly integrate into your workflow, ensuring efficiency and creativity. These components are designed to be versatile and customizable, allowing you to tailor them to your specific needs. Whether you are building a new website from scratch or enhancing an existing one, our AI-generated components provide a solid foundation for your project.  </p>
          </motion.div>
          <button className="mt-10 lg:mt-20 bg-[#191c1f] border-2 border-[#28292b] text-white px-4 py-2 rounded-lg self-start text-sm flex items-center">
            View More Details
            <FaArrowCircleRight className="ml-2" />
          </button>
        </div>
        <div className="w-full lg:w-1/2 p-5 flex items-center justify-center">
          {isMobile ? (
            <img src="assets/bentobox.jpg" alt="BentoBox Preview" className="w-full h-auto lg:h-full rounded-xl" />
          ) : (
            <video ref={videoRef1} className="w-full h-auto lg:h-full rounded-xl" muted loop>
              <source src="assets/bentobox.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          )}
        </div>
      </motion.div>
      <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
        <motion.div
          ref={ref2}
          className="bg-[#121416] border-2 border-[#28292b] h-auto w-full md:w-1/2 rounded-3xl shadow-lg flex flex-col justify-between p-3 hover:shadow-[0_0_15px_5px_rgba(0,255,255,0.3)] transition-shadow duration-300 sm:cursor-none"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: isInView2 ? 1 : 0, x: isInView2 ? 0 : -50 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          onMouseEnter={() => handleMouseEnter('You', videoRef2)}
          onMouseLeave={() => handleMouseLeave(videoRef2)}
        >
          <div className="p-5">
            {isMobile ? (
              <img src="assets/video3.jpg" alt="Video 3 Preview" className="w-full h-auto max-h-[400px] md:max-h-full rounded-xl" />
            ) : (
              <video ref={videoRef2} className="w-full h-auto max-h-[400px] md:max-h-full rounded-xl" muted loop>
                <source src="assets/video3.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )}
          </div>
          <motion.div
            className="p-5 flex flex-col justify-between flex-grow"
            initial="hidden"
            animate={isInView2 ? 'visible' : 'hidden'}
            variants={fadeInUp}
            custom={0}
          >
            <div>
              <h2 className="text-2xl font-semibold text-gray-100 mb-2">AI-Powered Tools for Modern Web Designers</h2>
              <p className="text-gray-400 mt-3">Leverage advanced AI tools to create stunning web components effortlessly, enabling a faster, smarter design process.</p>
            </div>
            <button className="mt-10 bg-[#191c1f] border-2 border-[#28292b] text-white px-4 py-2 rounded-lg self-start text-sm flex items-center">
              View More Details
              <FaArrowCircleRight className="ml-2" />
            </button>
          </motion.div>
        </motion.div>
        <motion.div
          ref={ref3}
          className="bg-[#121416] border-2 border-[#28292b] h-auto w-full md:w-1/2 rounded-3xl shadow-lg flex flex-col justify-between p-3 hover:shadow-[0_0_15px_5px_rgba(0,255,255,0.3)] transition-shadow duration-300 sm:cursor-none"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: isInView3 ? 1 : 0, x: isInView3 ? 0 : 50 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          onMouseEnter={() => handleMouseEnter('You', videoRef3)}
          onMouseLeave={() => handleMouseLeave(videoRef3)}
        >
          <div className="p-5">
            {isMobile ? (
              <img src="assets/first.jpg" alt="First Preview" className="w-full h-auto max-h-[400px] md:max-h-full rounded-xl" />
            ) : (
              <video ref={videoRef3} className="w-full h-auto max-h-[400px] md:max-h-full rounded-xl" muted loop>
                <source src="assets/first.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )}
          </div>
          <motion.div
            className="p-5 flex flex-col justify-between flex-grow"
            initial="hidden"
            animate={isInView3 ? 'visible' : 'hidden'}
            variants={fadeInUp}
            custom={0}
          >
            <div>
              <h2 className="text-2xl font-semibold text-gray-100 mb-2">Seamless Integration for Optimal Performance</h2>
              <p className="text-gray-400 mt-3">Experience the ease of integrating AI-generated components into your projects, ensuring top performance and design consistency.</p>
            </div>
            <button className="mt-10 bg-[#191c1f] border-2 border-[#28292b] text-white px-4 py-2 rounded-lg self-start text-sm flex items-center">
              View More Details
              <FaArrowCircleRight className="ml-2" />
            </button>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default BentoBox;
