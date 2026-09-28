namespace HuluBeje.Web.Models;

public class SupportSession
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid? CustomerId { get; set; }

    public bool IsGuest { get; set; } = true;

    public string Title { get; set; } = "New conversation";

    public string Status { get; set; } = "ACTIVE";

    public DateTimeOffset StartedAt { get; set; } =
        DateTimeOffset.UtcNow;

    public DateTimeOffset LastActivity { get; set; } =
        DateTimeOffset.UtcNow;

    public DateTimeOffset ExpiresAt { get; set; } =
        DateTimeOffset.UtcNow.AddDays(30);

    public List<ChatMessage> Messages { get; set; } = new();
}