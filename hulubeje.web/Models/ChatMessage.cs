namespace HuluBeje.Web.Models;

public class ChatMessage
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid SessionId { get; set; }

    public string Role { get; set; } = "";

    public string Content { get; set; } = "";

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}