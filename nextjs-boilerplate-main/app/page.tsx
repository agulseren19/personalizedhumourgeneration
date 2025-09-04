import Link from "next/link";
import BackgroundWrapper from "./Components/BeamBackground";
import { Gamepad2, Brain, Users, Cloud, Target, Zap, Shield, TrendingUp, Rocket } from 'lucide-react';

export default function Home() {
  return (
    <main className="w-full">
      <BackgroundWrapper>
        {/* CAH Game Hero Section */}
        <section id="cah-hero" className="pt-32 pb-20 px-6 min-h-screen flex items-center">
          <div className="max-w-5xl mx-auto text-center">
            <div className="mb-10">
              <Gamepad2 size={88} className="mx-auto text-accent-turquoise" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-text-primary mb-10 leading-tight">
              Cards Against Humanity
            </h1>
            <p className="text-lg md:text-xl text-text-secondary mb-16 max-w-2xl mx-auto leading-relaxed">
              Experience the future of comedy with AI-powered humor generation. 
              Our system learns your preferences and creates personalized jokes just for you.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-20">
              <Link 
                href="/cah"
                className="bg-gradient-to-r from-accent-pink to-accent-turquoise text-background-cream font-bold py-4 px-8 rounded-lg hover:from-accent-pink/90 hover:to-accent-turquoise/90 transition-all transform hover:scale-102 shadow-lg text-base"
              >
                <Gamepad2 className="inline mr-2" size={18} />
                Start Playing
              </Link>
            </div>

            {/* Features Grid - Made less perfect */}
            <div className="grid md:grid-cols-3 gap-8 text-center mb-20">
              <div className="bg-text-primary/8 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/15 hover:bg-text-primary/12 transition-all transform hover:translate-y-[-2px]">
                <div className="mb-3">
                  <Brain size={44} className="mx-auto text-accent-turquoise" />
                </div>
                <h3 className="font-semibold text-text-primary mb-2 text-lg">Smart Learning</h3>
                <p className="text-text-secondary text-sm">AI learns your humor preferences through feedback and adapts to your style</p>
              </div>
              <div className="bg-text-primary/8 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/15 hover:bg-text-primary/12 transition-all transform hover:translate-y-[-2px]">
                <div className="mb-3">
                  <Users size={44} className="mx-auto text-accent-pink" />
                </div>
                <h3 className="font-semibold text-text-primary mb-2 text-lg">Multiple Personas</h3>
                <p className="text-text-secondary text-sm">Choose from diverse AI comedy personalities with unique humor styles</p>
              </div>
              <div className="bg-text-primary/8 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/15 hover:bg-text-primary/12 transition-all transform hover:translate-y-[-2px]">
                <div className="mb-3">
                  <Cloud size={44} className="mx-auto text-accent-turquoise" />
                </div>
                <h3 className="font-semibold text-text-primary mb-3 text-lg">CrewAI Powered</h3>
                <p className="text-text-secondary text-sm">Multi-agent AI orchestration for intelligent humor generation</p>
              </div>
            </div>

            {/* Additional CAH Features - Less grid-like */}
            <div className="mt-20">
              <h2 className="text-3xl font-bold text-text-primary mb-12">Why Choose AI CAH?</h2>
              <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
                <div className="bg-text-primary/4 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/8 text-left">
                  <div className="mb-3">
                    <Target size={32} className="text-accent-pink" />
                  </div>
                  <h3 className="font-semibold text-text-primary mb-2 text-base">Personalized Comedy</h3>
                  <p className="text-text-secondary text-sm">Every joke is tailored to your sense of humor based on your feedback and preferences</p>
                </div>
                <div className="bg-text-primary/4 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/8 text-left">
                  <div className="mb-3">
                    <Zap size={32} className="text-accent-turquoise" />
                  </div>
                  <h3 className="font-semibold text-text-primary mb-2 text-base">Real-time Generation</h3>
                  <p className="text-text-secondary text-sm">Generate fresh, contextual humor instantly with multiple AI models working together</p>
                </div>
                <div className="bg-text-primary/4 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/8 text-left">
                  <div className="mb-3">
                    <Shield size={32} className="text-accent-pink" />
                  </div>
                  <h3 className="font-semibold text-text-primary mb-2 text-base">Content Safety</h3>
                  <p className="text-text-secondary text-sm">Built-in content filtering ensures appropriate humor for your audience</p>
                </div>
                <div className="bg-text-primary/4 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/8 text-left">
                  <div className="mb-3">
                    <TrendingUp size={32} className="text-accent-turquoise" />
                  </div>
                  <h3 className="font-semibold text-text-primary mb-2 text-base">Analytics Dashboard</h3>
                  <p className="text-text-secondary text-sm">Track your humor preferences and see how the AI learns your style over time</p>
                </div>
              </div>
            </div>

            {/* How to Play Section */}
            <div className="mt-20">
              <h2 className="text-3xl font-bold text-text-primary mb-8">How to Play</h2>
              <div className="bg-text-primary/4 backdrop-blur-sm rounded-lg p-6 border border-text-secondary/8 text-left max-w-3xl mx-auto">
                <div className="text-text-secondary leading-relaxed space-y-3 text-sm">
                  <p>
                    There are three modes in the game: <strong className="text-text-primary">White Card Generation</strong>, <strong className="text-text-primary">Black Card Generation</strong>, and the <strong className="text-text-primary">Multiplayer Game</strong>.
                  </p>
                  
                  <p>
                    Start by trying <strong className="text-text-primary">White Card Generation</strong>. In this mode, you'll rate or pick cards to help the system learn what kind of humour you enjoy. You can also explore <strong className="text-text-primary">Black Card Generation</strong> to create personalized black prompt cards based on your preferences.
                  </p>
                  
                  <p>
                    Once you've trained the system a bit, you're ready for the <strong className="text-text-primary">Multiplayer Game</strong>. In each round, every player receives a hand of white cards (responses), and one black card (a question or fill-in-the-blank prompt) is shown. Players choose the funniest or most fitting white card from their hand, and the judge picks the winner of the round.
                  </p>
                  
                  <p>
                    To join a multiplayer game, make sure you <strong className="text-text-primary">sign in first</strong>.
                  </p>
                  
                  <p>
                    You can browse all available humour styles under the <strong className="text-text-primary">AI Comedians</strong> section, and track your card picks, ratings, and evolving preferences in the <strong className="text-text-primary">Your Learning</strong> tab.
                  </p>

                  <p>
                  This game includes safety filters to reduce offensive or inappropriate content. However, due to the satirical nature of Cards Against Humanity, some users may find it inappropriate. Please proceed accordingly.
                  </p>
                </div>
              </div>
            </div>

            {/* Call to Action - Less perfect */}
            <div className="mt-20 mb-20">
              <div className="bg-gradient-to-r from-accent-pink/15 to-accent-turquoise/15 backdrop-blur-sm rounded-xl p-8 border border-text-secondary/15">
                <h2 className="text-2xl md:text-3xl font-bold text-text-primary mb-4">Ready to Laugh?</h2>
                <p className="text-lg text-text-secondary mb-6">
                  Join the AI comedy revolution and experience personalized humor like never before
                </p>
                <Link 
                  href="/cah"
                  className="inline-block bg-gradient-to-r from-accent-pink to-accent-turquoise text-background-cream font-bold py-3 px-6 rounded-lg hover:from-accent-pink/90 hover:to-accent-turquoise/90 transition-all transform hover:scale-102 shadow-lg text-base"
                >
                  <Rocket className="inline mr-2" size={18} />
                  Get Started Now
                </Link>
              </div>
            </div>
          </div>
        </section>
      </BackgroundWrapper>
    </main>
  );
}
