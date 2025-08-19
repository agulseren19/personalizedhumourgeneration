"use client";  // Add this directive to indicate it's a client component

import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';

const DotGridBackground = ({ containerHeight, containerWidth, lineWidth = 1.5, beamWidth = 200 }: { containerHeight: number; containerWidth: number; lineWidth?: number; beamWidth?: number }) => {
  const defaultDotSize = 30; // Default size of the dots
  const defaultGridSpacing = 400; // Default grid spacing
  const linePadding = 10; // Padding around the dots for the lines
  const topOffset = 300 ; // Offset from the top

  const [dotSize, setDotSize] = useState(defaultDotSize);
  const [gridSpacing, setGridSpacing] = useState(defaultGridSpacing);
  const [dots, setDots] = useState([]);

  useEffect(() => {
    const updateForMobile = () => {
      if (window.innerWidth <= 768) { // Adjust based on your mobile breakpoint
        setDotSize(defaultDotSize / 3); // adjust to change circle size on mobile
        setGridSpacing(defaultGridSpacing / 3);// adjust to change grid size on mobile
      } else {
        setDotSize(defaultDotSize);
        setGridSpacing(defaultGridSpacing);
      }
    };

    updateForMobile();
    window.addEventListener('resize', updateForMobile);

    return () => {
      window.removeEventListener('resize', updateForMobile);
    };
  }, []);

  const generateDots = () => {
    if (!containerHeight || !containerWidth) return;

    const newDots = [];
    const numColumns = Math.ceil(containerWidth / gridSpacing) + 1;
    const numRows = Math.ceil((containerHeight - topOffset) / gridSpacing) + 1;
    const offsetX = (containerWidth - (numColumns - 1) * gridSpacing) / 2;

    for (let col = 0; col < numColumns; col++) {
      for (let row = 0; row < numRows; row++) {
        const x = col * gridSpacing + offsetX;
        const y = row * gridSpacing + topOffset;
        newDots.push({ x, y });
      }
    }
    setDots(newDots);
  };

  useEffect(() => {
    generateDots();
  }, [containerHeight, containerWidth, dotSize, gridSpacing]);

  const beamDuration = 2; // Duration of the beam animation
  const beamDelay = 10; // Minimum delay between beam animations

  const getRandomDelay = () => Math.random() * beamDelay; // Random delay between 0 and beamDelay seconds

  return (
    <div className="absolute top-0 left-0 w-full h-full overflow-hidden">
      {dots.map((dot, index) => (
        <div key={index}>
          {/* Render dot */}
          <div
            className="absolute rounded-full border-2"
            style={{
              width: `${dotSize}px`,
              height: `${dotSize}px`,
              left: `${dot.x}px`,
              top: `${dot.y}px`,
              background: 'linear-gradient(145deg, #1e293b, #0f172a)',
              borderColor: '#33313d',
            }}
          />
          {/* Render line to the right neighbor */}
          {dot.x + gridSpacing < containerWidth ? (
            <div
              className="absolute bg-slate-800"
              style={{
                width: `${gridSpacing - dotSize - 2 * linePadding}px`,
                height: `${lineWidth}px`,
                left: `${dot.x + dotSize + linePadding}px`,
                top: `${dot.y + dotSize / 2 - lineWidth / 2}px`,
                overflow: 'hidden',
              }}
            >
              <motion.div
                className="absolute top-0 left-0 h-full"
                style={{
                  width: `${beamWidth}px`,
                  background: 'linear-gradient(90deg, transparent, teal, transparent)',
                  opacity: 0.75,
                }}
                initial={{ x: '-200px' }}
                animate={{ x: `calc(${gridSpacing - dotSize - 2 * linePadding}px + ${beamWidth}px)` }}
                transition={{
                  repeat: Infinity,
                  duration: beamDuration,
                  repeatDelay: getRandomDelay(),
                  ease: 'linear',
                  delay: getRandomDelay(),
                }}
              />
            </div>
          ) : (
            <div
              className="absolute bg-slate-800"
              style={{
                width: `${containerWidth - dot.x - dotSize - linePadding}px`,
                height: `${lineWidth}px`,
                left: `${dot.x + dotSize + linePadding}px`,
                top: `${dot.y + dotSize / 2 - lineWidth / 2}px`,
                overflow: 'hidden',
              }}
            >
              <motion.div
                className="absolute top-0 left-0 h-full"
                style={{
                  width: `${beamWidth}px`,
                  background: 'linear-gradient(90deg, transparent, teal, transparent)',
                  opacity: 0.75,
                }}
                initial={{ x: '-200px' }}
                animate={{ x: `calc(${containerWidth - dot.x - dotSize - linePadding}px + ${beamWidth}px)` }}
                transition={{
                  repeat: Infinity,
                  duration: beamDuration,
                  repeatDelay: getRandomDelay(),
                  ease: 'linear',
                  delay: getRandomDelay(),
                }}
              />
            </div>
          )}
          {/* Render line to the bottom neighbor */}
          {dot.y + gridSpacing < containerHeight ? (
            <div
              className="absolute bg-slate-800"
              style={{
                width: `${lineWidth}px`,
                height: `${gridSpacing - dotSize - 2 * linePadding}px`,
                left: `${dot.x + dotSize / 2 - lineWidth / 2}px`,
                top: `${dot.y + dotSize + linePadding}px`,
                overflow: 'hidden',
              }}
            >
              <motion.div
                className="absolute top-0 left-0 w-full"
                style={{
                  height: `${beamWidth}px`,
                  background: 'linear-gradient(180deg, transparent, teal, transparent)',
                  opacity: 0.75,
                }}
                initial={{ y: '-200px' }}
                animate={{ y: `calc(${gridSpacing - dotSize - 2 * linePadding}px + ${beamWidth}px)` }}
                transition={{
                  repeat: Infinity,
                  duration: beamDuration,
                  repeatDelay: getRandomDelay(),
                  ease: 'linear',
                  delay: getRandomDelay(),
                }}
              />
            </div>
          ) : (
            <div
              className="absolute bg-slate-800"
              style={{
                width: `${lineWidth}px`,
                height: `${containerHeight - dot.y - dotSize - linePadding}px`,
                left: `${dot.x + dotSize / 2 - lineWidth / 2}px`,
                top: `${dot.y + dotSize + linePadding}px`,
                overflow: 'hidden',
              }}
            >
              <motion.div
                className="absolute top-0 left-0 w-full"
                style={{
                  height: `${beamWidth}px`,
                  background: 'linear-gradient(180deg, transparent, teal, transparent)',
                  opacity: 0.75,
                }}
                initial={{ y: '-200px' }}
                animate={{ y: `calc(${containerHeight - dot.y - dotSize - linePadding}px + ${beamWidth}px)` }}
                transition={{
                  repeat: Infinity,
                  duration: beamDuration,
                  repeatDelay: getRandomDelay(),
                  ease: 'linear',
                  delay: getRandomDelay(),
                }}
              />
            </div>
          )}
          {/* Render line to the top neighbor */}
          <div
            className="absolute bg-slate-800"
            style={{
              width: `${lineWidth}px`,
              height: `${gridSpacing - dotSize - 2 * linePadding}px`,
              left: `${dot.x + dotSize / 2 - lineWidth / 2}px`,
              top: `${dot.y - gridSpacing + dotSize + linePadding}px`,
              overflow: 'hidden',
            }}
          >
            <motion.div
              className="absolute bottom-0 left-0 w-full"
              style={{
                height: `${beamWidth}px`,
                background: 'linear-gradient(0deg, transparent, teal, transparent)',
                opacity: 0.75,
              }}
              initial={{ y: '200px' }}
              animate={{ y: `calc(-${gridSpacing - dotSize - 2 * linePadding}px - ${beamWidth}px)` }}
              transition={{
                repeat: Infinity,
                duration: beamDuration,
                repeatDelay: getRandomDelay(),
                ease: 'linear',
                delay: getRandomDelay(),
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

// BackgroundWrapper Component
const BackgroundWrapper = ({ children }) => {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    return () => {
      window.removeEventListener('resize', updateDimensions);
    };
  }, []);

  return (
    <div className="relative w-full h-full min-h-screen bg-slate-900" ref={containerRef}>
      <DotGridBackground containerHeight={dimensions.height} containerWidth={dimensions.width} />
      <div className="relative z-10">{children}</div>
    </div>
  );
};
export default BackgroundWrapper;

// FAQAccordion Component
