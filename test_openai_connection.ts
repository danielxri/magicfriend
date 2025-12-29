import "dotenv/config";
import OpenAI from "openai";

async function testOpenAI() {
    console.log("Testing OpenAI connection...");
    console.log("API Key present:", !!process.env.OPENAI_API_KEY);
    console.log("API Key format check:", process.env.OPENAI_API_KEY?.startsWith("sk-") ? "Valid prefix" : "Invalid prefix");

    const openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
    });

    try {
        console.log("Attempting to list models...");
        const models = await openai.models.list();
        console.log("Successfully connected! Found models:", models.data.length);

        const gpt52 = models.data.find(m => m.id === "gpt-5.2");
        console.log("GPT-5.2 available:", !!gpt52);

        const gptImage1 = models.data.find(m => m.id.includes("gpt-image") || m.id.includes("dall-e"));
        console.log("Image model available:", gptImage1?.id || "None found");

    } catch (error) {
        console.error("Connection failed:", error);
    }
}

testOpenAI();
