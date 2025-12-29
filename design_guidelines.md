# Design Guidelines: Kids' Imagination App

## Design Approach
**Reference-Based Approach** inspired by Duolingo's playful interaction patterns, Snapchat's camera-first interface, and YouTube Kids' child-friendly design principles. This experience-focused app prioritizes delight, safety, and intuitive interactions for young users.

## Core Design Principles
- **Magical & Playful**: Every interaction should feel like discovering something wonderful
- **Touch-First**: All controls sized for small fingers (minimum 44px touch targets)
- **Visual Feedback**: Immediate, animated responses to every action
- **Safety-First**: Clear parental controls and premium boundaries

## Typography System
- **Primary Font**: Fredoka (Google Fonts) - rounded, friendly, highly legible for kids
- **Hierarchy**:
  - Hero/App Title: 2.5rem (bold) desktop, 2rem mobile
  - Mode Headers: 1.75rem (semibold)
  - Button Text: 1.125rem (bold) - oversized for readability
  - Body/Instructions: 1rem (medium)
  - Helper Text: 0.875rem (regular)

## Layout & Spacing System
**Spacing Primitives**: Tailwind units of 4, 6, 8, 12, 16
- Component padding: p-6 to p-8
- Section gaps: gap-8 to gap-12
- Button padding: px-8 py-4 (extra generous for kids)
- Screen margins: px-4 mobile, px-8 desktop

## Primary Layout Structure

### Camera Interface (Both Modes)
- Full-viewport camera preview (100vh on mobile, 80vh desktop)
- Floating controls overlay with backdrop blur
- Large circular capture button centered bottom (96px diameter)
- Mode toggle prominent top-center
- Settings/parent controls tucked top-right (small, deliberate)

### Free Mode Flow
1. **Capture Screen**: Camera viewfinder with playful frame overlay
2. **Processing State**: Full-screen animated loader (bouncing character graphics)
3. **Result Screen**: Generated image fills viewport with "Try Again" and "Save" buttons as floating cards at bottom

### Premium Mode Interface
- **Split Layout (Desktop)**: Generated avatar left 50%, conversation controls right 50%
- **Stacked Layout (Mobile)**: Avatar full-width top, controls fixed bottom sheet
- **Avatar Display**: Contained in rounded container with subtle glow effect
- **Recording Button**: Extra-large pulsing circle when active (120px), visual waveform animation
- **Conversation Bubbles**: Large, rounded speech bubbles with clear kid/character distinction

## Component Library

### Navigation
- **Mode Switcher**: Large toggle pill (Free/Premium) with sliding indicator, centered top
- **Parent Lock**: Small lock icon top-right requiring hold gesture to access

### Interactive Elements
- **Primary Buttons**: 
  - Extra rounded (rounded-full or rounded-3xl)
  - Large text (text-lg)
  - Generous padding (px-8 py-4)
  - When on images: backdrop-blur-lg with semi-transparent backgrounds
- **Camera Capture**: Circular button with animated ring on press
- **Recording Button**: Pulsing circle with concentric wave animations when active
- **Icon Buttons**: 56px minimum, clear single-purpose icons from Heroicons

### Cards & Containers
- **Result Cards**: Generous rounded corners (rounded-2xl to rounded-3xl)
- **Avatar Container**: Rounded frame with playful border treatment
- **Premium Upsell Card**: Centered card with sparkle/star graphics, clear benefit bullets

### Form Elements
- **Parent Email Input** (for premium): Large text input (h-14), rounded-xl
- **Upgrade Button**: Hero-sized, animated with particle effects on hover

### Visual Feedback
- **Loading States**: Animated character illustrations (not spinners)
- **Success Animations**: Confetti or star burst effects
- **Error States**: Friendly character shrug with simple retry option
- **Audio Indicators**: Animated waveform bars during speech

## Screen Specifications

### Landing/Home Screen
- Hero: Full-viewport example transformation (before/after slider)
- CTA: "Start Creating Magic" mega-button
- Two-column feature showcase (desktop): Free mode features | Premium mode features
- Gentle scroll indicator (bounce animation)

### Mode Selection/Onboarding
- Two large cards side-by-side (desktop) or stacked (mobile)
- Each card shows mode preview with key feature bullets
- Visual distinction through illustrations (basic face vs talking character)

### Premium Paywall
- Full-screen modal overlay
- Animated character demonstration at top
- 3-column benefit grid (desktop), stacked (mobile)
- Price display with trial callout
- Parent verification step before purchase flow

## Animations (Strategic Use Only)
- **Entry**: Gentle scale + fade for modals (0.2s)
- **Capture**: Flash + scale on photo snap
- **Processing**: Looping character bounce animation
- **Success**: One-time confetti burst
- **Recording**: Continuous pulse during audio capture
- **Avatar Speech**: Mouth/face animation synced to audio (handled by HeyGen)

## Accessibility for Kids
- Consistent 44px+ touch targets throughout
- High contrast ratios (WCAG AAA where possible)
- Simple, icon-supported text labels
- Screen reader support for parent controls
- No time-based interactions kids can't complete
- Clear visual state changes (not just color)

## Images
**Hero Section**: Full-width before/after transformation showcase
- Before: Child holding stuffed animal/toy
- After: Same toy with cute animated face
- Implementation: Side-by-side slider or auto-animate between states

**Feature Illustrations**: Custom character graphics for each feature benefit (friendly objects with faces)

**Avatar Placeholder**: Cheerful default character when no photo loaded

This design creates an immersive, magical experience where every interaction delights young users while maintaining clear boundaries and controls for parents.