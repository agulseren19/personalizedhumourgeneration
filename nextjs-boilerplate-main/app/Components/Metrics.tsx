// components/Metrics.js
"use client";
import React, { useEffect, useState } from 'react';
import { motion, useSpring, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import Lottie from 'react-lottie';
import increaseAnimationData from '../assets/increase.json';
import decreaseAnimationData from '../assets/decrease.json';

const metrics = [
  { value: '37k', subtitle: 'AI-generated components created to enhance your web projects.', tooltip: 'Component generation', change: { direction: 'increase', percentage: 12 } },
  { value: '200', subtitle: 'Satisfied customers who trust our AI solutions.', tooltip: 'Customer satisfaction', change: { direction: 'decrease', percentage: 5 } },
  { value: '15k', subtitle: 'Projects completed using our AI-generated components.', tooltip: 'Successful projects', change: { direction: 'increase', percentage: 8 } },
  { value: '5k', subtitle: 'Support tickets resolved by our dedicated team.', tooltip: 'Efficient support', change: { direction: 'decrease', percentage: 3 } },
];

const Counter = ({ from, to, isK, start }) => {
  const [count, setCount] = useState(from);
  const spring = useSpring(from, { stiffness: 30, damping: 15 });

  useEffect(() => {
    if (start) {
      spring.set(to);
      spring.onChange((value) => setCount(Math.round(value)));
    }
  }, [spring, to, start]);

  return <motion.span>{isK ? (count / 1000).toFixed(1) + 'k' : count}</motion.span>;
};

const Metrics = () => {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });
  const [hoveredMetric, setHoveredMetric] = useState(null);
  const [tooltipX, setTooltipX] = useState(0);
  const [isStopped, setIsStopped] = useState(true);
  const [isPaused, setIsPaused] = useState(true);

  useEffect(() => {
    if (hoveredMetric !== null) {
      setIsStopped(false);
      setIsPaused(false);
    } else {
      setIsStopped(true);
      setIsPaused(true);
    }
  }, [hoveredMetric]);

  const handleMouseMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;

    if (mouseX < rect.width / 3) {
      setTooltipX(-30);
    } else if (mouseX > (2 * rect.width) / 3) {
      setTooltipX(30);
    } else {
      setTooltipX(0);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.4 // Increase this value to make the appearance more gradual
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 50 }, // Increase y value for a more dramatic entrance
    visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 60, damping: 15, duration: 0.5 } } // Adjust duration for smoothness
  };

  const defaultOptions = (animationData) => ({
    loop: false,
    autoplay: false,
    animationData: animationData,
    rendererSettings: {
      preserveAspectRatio: 'xMidYMid slice'
    }
  });

  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: i => ({
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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 sm:pt-72">
      <div ref={ref} className="mx-auto max-w-5xl text-center mb-12">
        <motion.h1
          className="text-3xl font-bold tracking-tight text-white sm:text-5xl text-center"
          initial="hidden"
          animate={inView ? 'visible' : 'hidden'}
          variants={fadeInUp}
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
          {' '}Empowering Your Web Development
        </motion.h1>
        <motion.p
          className="mt-10 text-lg leading-8 text-gray-200 max-w-3xl mx-auto mb-20"
          initial="hidden"
          animate={inView ? 'visible' : 'hidden'}
          variants={fadeInUp}
        >
          GenUI integrates cutting-edge AI to provide you with powerful tools for generating web components, accelerating your development process, and enhancing your projects.
        </motion.p>
      </div>

      <motion.div
        ref={ref}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
        variants={containerVariants}
        initial="hidden"
        animate={inView ? 'visible' : 'hidden'}
      >
        {metrics.map((metric, index) => {
          const value = parseInt(metric.value.replace(/[^0-9]/g, '')) * (metric.value.includes('k') ? 1000 : 1);
          const isK = metric.value.includes('k');
          const changeText = metric.change.direction === 'increase'
            ? `Increased by ${metric.change.percentage}%`
            : `Decreased by ${metric.change.percentage}%`;
          const changeAnimation = metric.change.direction === 'increase' ? increaseAnimationData : decreaseAnimationData;
          const tooltipClass = metric.change.direction === 'increase' ? 'bg-green-100 border-green-500' : 'bg-red-100 border-red-500';

          return (
            <motion.div
              key={index}
              className="bg-[#121416] border-2 border-[#28292b] rounded-lg flex flex-col justify-top items-center p-6 h-[150px] relative "
              variants={itemVariants}
              onHoverStart={() => setHoveredMetric(index)}
              onHoverEnd={() => setHoveredMetric(null)}
              onMouseMove={handleMouseMove}
            >
              <div className="text-white text-3xl font-semibold">
                <Counter from={0} to={value} isK={isK} start={inView} />
              </div>
              <div className="text-gray-300 text-sm font-light mt-2 text-center">
                {metric.subtitle}
              </div>
              <AnimatePresence>
                {hoveredMetric === index && (
                  <motion.div
                    className={`absolute top-0 transform -translate-y-20 h-10 ${tooltipClass} rounded-lg flex items-center justify-center px-4 py-2 border`}
                    initial={{ opacity: 0, y: -10, scale: 0.8, x: tooltipX }}
                    animate={{
                      opacity: 1,
                      y: [0, -60],
                      x: 0,
                      scale: 1,
                      transition: {
                        y: { type: 'spring', stiffness: 500, damping: 20 },
                        x: { type: 'spring', stiffness: 100, damping: 10 },
                        opacity: { duration: 0.2 },
                      },
                    }}
                    exit={{ opacity: 0, y: -10, scale: 0.8 }}
                  >
                    <span className="mr-2">{changeText}</span>
                    <Lottie
                      options={defaultOptions(changeAnimation)}
                      height={30}
                      width={30}
                      isStopped={isStopped}
                      isPaused={isPaused}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
};

export default Metrics;
