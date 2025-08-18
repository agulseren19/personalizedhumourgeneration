// src/components/Navbar.js
"use client";
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { useUser } from '../contexts/UserContext';
import AuthModal from './auth/AuthModal';

const Navbar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const { user, logout } = useUser();

  const handleMobileMenuToggle = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const mobileMenuVariants = {
    hidden: { opacity: 0, height: 0 },
    visible: { opacity: 1, height: 'auto', transition: { staggerChildren: 0.3 } },
  };

  const menuItemVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: { opacity: 1, y: 0 },
  };

  const navbarVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, when: "beforeChildren", staggerChildren: 0.2 },
    },
  };

  const linkVariants = {
    hidden: { opacity: 0, x: -50 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.5 } },
  };

  useEffect(() => {
    // On load animation for the navbar
    document.querySelector('nav').classList.add('animate-onload');
  }, []);

  const handleHomeScroll = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <motion.nav
      className="fixed top-0 w-full z-50 sm:p-6 p-3 bg-gray-900/20 backdrop-blur-sm"
      initial="hidden"
      animate="visible"
      variants={navbarVariants}
    >
      <motion.div
        className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-8 bg-gray-900/95 backdrop-blur-md shadow-lg py-4 rounded-xl border border-white/10"
        variants={navbarVariants}
      >
        <div className="flex items-center justify-between h-12">
          <div className="flex items-center flex-shrink-0">
            <Link href="/" className="cursor-pointer">
              <div className="text-white font-bold text-lg flex items-center space-x-1 whitespace-nowrap">
                <span>🃏 AI Cards Against Humanity</span>
              </div>
            </Link>
          </div>
          <div className="hidden md:flex justify-center flex-1 mx-8">
            <ul className="flex items-center space-x-8">
              <motion.li variants={linkVariants}>
                <Link href="/" className="text-gray-200 hover:text-teal-400 cursor-pointer transition-colors">
                  Home
                </Link>
              </motion.li>
              <motion.li variants={linkVariants}>
                <Link href="/cah" className="text-gray-200 hover:text-teal-400 cursor-pointer transition-colors">
                  Game
                </Link>
              </motion.li>
              <motion.li variants={linkVariants}>
                <Link href="/cah?tab=analytics" className="text-gray-200 hover:text-teal-400 cursor-pointer transition-colors">
                  Analytics
                </Link>
              </motion.li>
            </ul>
          </div>
          <motion.div className="hidden md:block" variants={linkVariants}>
            {user ? (
              <div className="flex items-center space-x-4">
                <span className="text-gray-300 text-sm">Welcome, {user.email}</span>
                <button
                  onClick={logout}
                  className="text-gray-300 hover:text-red-400 transition-colors px-3 py-2 rounded-lg hover:bg-white/10"
                >
                  Logout
                </button>
                <Link href="/cah" className="bg-teal-400 text-white px-6 py-3 rounded-xl hover:bg-teal-500 transition-all duration-200 font-medium shadow-md hover:shadow-lg">
                  Play Game
                </Link>
              </div>
            ) : (
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="text-gray-300 hover:text-teal-400 transition-colors px-3 py-2 rounded-lg hover:bg-white/10"
                >
                  Sign In
                </button>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="bg-teal-400 text-white px-6 py-3 rounded-xl hover:bg-teal-500 transition-all duration-200 font-medium shadow-md hover:shadow-lg"
                >
                  Get Started
                </button>
              </div>
            )}
          </motion.div>
          <div className="md:hidden flex items-center">
            <button onClick={handleMobileMenuToggle} className="text-white">
              {isMobileMenuOpen ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
                </svg>
              )}
            </button>
          </div>
        </div>
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              initial="hidden"
              animate="visible"
              exit="hidden"
              variants={mobileMenuVariants}
              className="md:hidden mt-4"
            >
              <ul className="space-y-4">
                <motion.li variants={menuItemVariants}>
                  <Link href="/" className="text-gray-200 hover:text-teal-400 cursor-pointer transition-colors">
                    Home
                  </Link>
                </motion.li>
                <motion.li variants={menuItemVariants}>
                  <Link href="/cah" className="text-gray-200 hover:text-teal-400 cursor-pointer transition-colors">
                    Game
                  </Link>
                </motion.li>
                <motion.li variants={menuItemVariants}>
                  <Link href="/cah?tab=analytics" className="text-gray-200 hover:text-teal-400 cursor-pointer transition-colors">
                    Analytics
                  </Link>
                </motion.li>

              </ul>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
      
      {/* Auth Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
      />
    </motion.nav>
  );
};

export default Navbar;
