using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HuluBeje.Web.Services;

public class RagService
{
    private readonly HttpClient _httpClient;

    public RagService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }


    // =========================================================
    // TEXT CHAT
    // =========================================================

    public async Task<string> AskAsync(
        string message,
        List<ConversationHistoryMessage> conversationHistory)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            throw new ArgumentException(
                "Message cannot be empty.",
                nameof(message)
            );
        }

        var request = new ChatRequest
        {
            Message = message,
            ConversationHistory = conversationHistory
        };

        var response = await _httpClient.PostAsJsonAsync(
            "api/v1/chat",
            request
        );

        response.EnsureSuccessStatusCode();

        var result =
            await response.Content.ReadFromJsonAsync<ChatResponse>();

        return result?.Answer
            ?? "Sorry, I couldn't generate an answer.";
    }


    // =========================================================
    // VOICE CHAT
    // =========================================================

    public async Task<VoiceChatResponse> VoiceChatAsync(
        byte[] audioBytes,
        List<ConversationHistoryMessage> conversationHistory)
    {
        if (audioBytes is null || audioBytes.Length == 0)
        {
            throw new ArgumentException(
                "Audio cannot be empty.",
                nameof(audioBytes)
            );
        }

        var historyJson = JsonSerializer.Serialize(
            conversationHistory
        );

        using var form = new MultipartFormDataContent();

        using var audioContent =
            new ByteArrayContent(audioBytes);

        audioContent.Headers.ContentType =
            new MediaTypeHeaderValue("audio/wav");

        form.Add(
            audioContent,
            "audio",
            "voice.wav"
        );

        form.Add(
            new StringContent(historyJson),
            "conversation_history"
        );

        var response = await _httpClient.PostAsync(
            "api/v1/voice/chat",
            form
        );

        response.EnsureSuccessStatusCode();

        var result =
            await response.Content.ReadFromJsonAsync<VoiceChatResponse>();

        if (result is null)
        {
            throw new InvalidOperationException(
                "Voice API returned an empty response."
            );
        }

        return result;
    }


    // =========================================================
    // TEXT API MODELS
    // =========================================================

    private sealed class ChatRequest
    {
        public string Message { get; set; } = "";

        [JsonPropertyName("conversation_history")]
        public List<ConversationHistoryMessage> ConversationHistory
        {
            get;
            set;
        } = new();
    }


    private sealed class ChatResponse
    {
        public string Answer { get; set; } = "";
    }


    // =========================================================
    // VOICE API RESPONSE
    // =========================================================

    public sealed class VoiceChatResponse
    {
        public string Transcription { get; set; } = "";

        public string Answer { get; set; } = "";

        [JsonPropertyName("original_language")]
        public string OriginalLanguage { get; set; } = "";

        [JsonPropertyName("retrieval_query")]
        public string RetrievalQuery { get; set; } = "";
    }
}


// =============================================================
// SHARED CONVERSATION MESSAGE
// =============================================================

public class ConversationHistoryMessage
{
    public string Role { get; set; } = "";

    public string Content { get; set; } = "";
}