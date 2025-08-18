"use client";

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import CircleWave from './CircleWave';
import { GetStarted } from './Buttons/GetStarted';

const CTA = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  return (
    <div className="max-w-7xl mx-auto pb-20 p-3">
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: isInView ? 1 : 0, y: isInView ? 0 : 50 }}
        transition={{ duration: 0.5 }}
        className="relative isolate overflow-hidden bg-[#121416] border-2 border-[#28292b] rounded-xl"
      >
        <div className="absolute inset-0">
          <CircleWave className="absolute bottom-20 left-96" delayOffset={0.5} />
          <CircleWave className="absolute bottom-[650px] right-[200px] sm:bottom-[700px] sm:right-[600px]" delayOffset={1.5} />
        </div>
        <div className="relative px-6 py-24 sm:px-6 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Transform Your Web Development.
              <br />
              Start Using GenAI Today.
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-gray-300">
            Unlock the power of AI-generated web components with GenAI. Enhance your productivity, streamline your workflow, and elevate your projects to new heights.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <GetStarted name="Get Started" />
            </div>
          </div>
        </div>
        <svg
          viewBox="0 0 1024 1024"
          className="absolute left-1/2 top-1/2 -z-10 h-[64rem] w-[64rem] -translate-x-1/2 [mask-image:radial-gradient(closest-side,white,transparent)]"
          aria-hidden="true"
        >
          <circle cx={512} cy={512} r={512} fill="url(#8d958450-c69f-4251-94bc-4e091a323369)" fillOpacity="0.7" />
          <defs>
            <radialGradient id="8d958450-c69f-4251-94bc-4e091a323369">
              <stop stopColor="#00FFAA" />
              <stop offset={1} stopColor="#008080" />
            </radialGradient>
          </defs>
        </svg>
      </motion.div>
    </div>
  );
}

export default CTA;
