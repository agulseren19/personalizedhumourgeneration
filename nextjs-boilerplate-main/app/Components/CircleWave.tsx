import { motion } from 'framer-motion';

const CircleWave = ({ className, delayOffset = 0 }: { className?: string; delayOffset?: number }) => {
    const circleVariants = {
        hidden: { opacity: 0, scale: 0.5 },
        visible: (i: number) => ({
            opacity: [0, 0.3, 0],
            scale: [0.5, 1.5, 2],
            transition: {
                delay: i * 0.5 + delayOffset,
                duration: 5,
                repeat: Infinity,
                repeatType: 'loop',
                ease: 'easeInOut'
            }
        })
    };

    return (
        <div className={`relative flex justify-center items-center h-full w-full bg-transparent blur-xl ${className}`}>
            {[...Array(4)].map((_, i) => {
                const isFirstCircle = i === 0;
                return (
                    <motion.div
                        custom={i}
                        key={i}
                        className={`absolute rounded-full border border-teal-200 bg-teal-500 flex justify-center items-center ${isFirstCircle ? 'static' : ''}`}
                        style={{
                            width: isFirstCircle ? '80px' : `${60 + i * 45}px`,
                            height: isFirstCircle ? '80px' : `${60 + i * 45}px`,
                            opacity: 0.3
                        }}
                        variants={isFirstCircle ? {} : circleVariants}
                        initial="hidden"
                        animate="visible"
                    >
                    </motion.div>
                );
            })}
        </div>
    );
};

export default CircleWave;
