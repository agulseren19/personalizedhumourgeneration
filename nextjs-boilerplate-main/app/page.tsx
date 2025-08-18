import Link from "next/link";
import BackgroundWrapper from "./Components/BeamBackground";

export default function Home() {
  return (
    <main className="w-full">
      <BackgroundWrapper>
        {/* CAH Game Hero Section */}
        <section id="cah-hero" className="pt-40 pb-20 px-4 min-h-screen flex items-center">
          <div className="max-w-4xl mx-auto text-center">
            <div className="text-8xl mb-8">🃏</div>
            <h1 className="text-6xl md:text-8xl font-bold text-white mb-8">
              AI Cards Against Humanity
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Experience the future of comedy with AI-powered humor generation. 
              Our system learns your preferences and creates personalized jokes just for you.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-16">
              <Link 
                href="/cah"
                className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-5 px-10 rounded-xl hover:from-purple-700 hover:to-indigo-700 transition-all transform hover:scale-105 shadow-xl text-lg"
              >
                🎮 Start Playing
              </Link>
            </div>

            {/* Features Grid */}
            <div className="grid md:grid-cols-3 gap-8 text-center">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 hover:bg-white/15 transition-all">
                <div className="text-4xl mb-4">🧠</div>
                <h3 className="font-semibold text-white mb-3 text-xl">Smart Learning</h3>
                <p className="text-gray-300">AI learns your humor preferences through feedback and adapts to your style</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 hover:bg-white/15 transition-all">
                <div className="text-4xl mb-4">🎭</div>
                <h3 className="font-semibold text-white mb-3 text-xl">Multiple Personas</h3>
                <p className="text-gray-300">Choose from diverse AI comedy personalities with unique humor styles</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 hover:bg-white/15 transition-all">
                <div className="text-4xl mb-4">☁️</div>
                <h3 className="font-semibold text-white mb-3 text-xl">AWS Powered</h3>
                <p className="text-gray-300">Advanced vector embeddings and knowledge graphs for better humor</p>
              </div>
            </div>

            {/* Additional CAH Features */}
            <div className="mt-20">
              <h2 className="text-4xl font-bold text-white mb-12">Why Choose AI CAH?</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10">
                  <div className="text-3xl mb-4">🎯</div>
                  <h3 className="font-semibold text-white mb-3 text-lg">Personalized Comedy</h3>
                  <p className="text-gray-400">Every joke is tailored to your sense of humor based on your feedback and preferences</p>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10">
                  <div className="text-3xl mb-4">⚡</div>
                  <h3 className="font-semibold text-white mb-3 text-lg">Real-time Generation</h3>
                  <p className="text-gray-400">Generate fresh, contextual humor instantly with multiple AI models working together</p>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10">
                  <div className="text-3xl mb-4">🛡️</div>
                  <h3 className="font-semibold text-white mb-3 text-lg">Content Safety</h3>
                  <p className="text-gray-400">Built-in content filtering ensures appropriate humor for your audience</p>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10">
                  <div className="text-3xl mb-4">📈</div>
                  <h3 className="font-semibold text-white mb-3 text-lg">Analytics Dashboard</h3>
                  <p className="text-gray-400">Track your humor preferences and see how the AI learns your style over time</p>
                </div>
              </div>
            </div>

            {/* Call to Action */}
            <div className="mt-20 mb-20">
              <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 backdrop-blur-sm rounded-2xl p-12 border border-white/20">
                <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">Ready to Laugh?</h2>
                <p className="text-xl text-gray-300 mb-8">
                  Join the AI comedy revolution and experience personalized humor like never before
                </p>
                <Link 
                  href="/cah"
                  className="inline-block bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-4 px-8 rounded-xl hover:from-purple-700 hover:to-indigo-700 transition-all transform hover:scale-105 shadow-xl text-lg"
                >
                  🚀 Get Started Now
                </Link>
              </div>
            </div>
          </div>
        </section>
      </BackgroundWrapper>
    </main>
  );
}
