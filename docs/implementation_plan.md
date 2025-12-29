# Feature: Context-Aware Personality Generation

## Goal Description
The user wants the generated character's personality and name to be "creatively linked to the object" in the photo. Currently, the system generates a generic magical personality without knowing what the object is.

## User Review Required
> [!NOTE]
> This change switches the model for `/api/character-card` from `gpt-5.2` (Text-Only) to `gpt-4o` (Vision). This allows the AI to "see" the object.

## Proposed Changes
### Backend (Node.js)
#### [MODIFY] [server/routes.ts](file:///home/danwils/magicfriend/Imagine-Friend-2/server/routes.ts)
- **Locate endpoint**: `/api/character-card`.
- **Retrieve Image**: Get the `originalImageUrl` from the session storage.
- **Keep Model**: `gpt-5.2` (Confirmed Multimodal).
- **Update Payload**:
    - Add the image to the `user` message using `{ type: "image_url", image_url: { url: ... } }`.
    - Update `system` prompt to explicitly instruct the AI to analyze the visual object.
    - **Prompt Instruction**: "Analyze the image. Identify the object (e.g. a stapler, a plant, a shoe). Create a character based on this object. If it's a shoe, maybe they love running but get tired. If it's a clock, maybe they are anxious about time."

## Verification Plan
### Manual Verification
1.  **Restart Server**: `npm run dev`.
2.  **Take Photo**: Photograph a specific object (e.g., a Coffee Mug).
3.  **Check Result**: Verify the generated name and bio relate to the object (e.g., "Muggy the Warm", "Loves holding hot drinks").
