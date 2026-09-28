namespace HuluBeje.Web.Models;

public class ChatConversation
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public string Title { get; set; } = "New chat";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public List<ChatMessage> Messages { get; set; } = new();
}