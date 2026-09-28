using HuluBeje.Web.Models;

namespace HuluBeje.Web.State;

public class ChatState
{
    public List<ChatConversation> Conversations { get; private set; } = new();

    public ChatConversation? ActiveConversation { get; private set; }

    public ChatState()
    {
        CreateNewChat();
    }

    public void CreateNewChat()
    {
        var conversation = new ChatConversation();

        Conversations.Insert(0, conversation);

        ActiveConversation = conversation;
    }

    public void SelectConversation(Guid conversationId)
    {
        var conversation = Conversations
            .FirstOrDefault(x => x.Id == conversationId);

        if (conversation is not null)
        {
            ActiveConversation = conversation;
        }
    }
}