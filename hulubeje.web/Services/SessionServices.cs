using HuluBeje.Web.Models;

namespace HuluBeje.Web.Services;

public class SessionService
{
    private Customer? _customer;

    private readonly List<SupportSession> _sessions = new();

    private SupportSession? _activeSession;

    public Customer? CurrentCustomer => _customer;

    public IReadOnlyList<SupportSession> Sessions => _sessions;

    public SupportSession? ActiveSession => _activeSession;

    public bool IsGuest =>
        _activeSession?.IsGuest ?? true;


    // =====================================================
    // CREATE GUEST SESSION
    // =====================================================

    public SupportSession CreateGuestSession()
    {
        var session = new SupportSession
        {
            IsGuest = true,
            StartedAt = DateTimeOffset.UtcNow,
            LastActivity = DateTimeOffset.UtcNow,
            ExpiresAt = DateTimeOffset.UtcNow.AddDays(30)
        };

        _sessions.Insert(0, session);

        _activeSession = session;

        return session;
    }


    // =====================================================
    // CREATE CUSTOMER
    // =====================================================

    public Customer CreateCustomer(
        string name,
        string contact)
    {
        _customer = new Customer
        {
            Name = name.Trim(),
            Contact = contact.Trim()
        };

        return _customer;
    }


    // =====================================================
    // CREATE CUSTOMER SESSION
    // =====================================================

    public SupportSession CreateCustomerSession()
    {
        if (_customer is null)
        {
            throw new InvalidOperationException(
                "A customer must exist before creating a customer session."
            );
        }

        var session = new SupportSession
        {
            CustomerId = _customer.Id,
            IsGuest = false,
            StartedAt = DateTimeOffset.UtcNow,
            LastActivity = DateTimeOffset.UtcNow,
            ExpiresAt = DateTimeOffset.UtcNow.AddDays(30)
        };

        _sessions.Insert(0, session);

        _activeSession = session;

        return session;
    }


    // =====================================================
    // SELECT SESSION
    // =====================================================

    public void SelectSession(Guid sessionId)
    {
        var session = _sessions.FirstOrDefault(
            x => x.Id == sessionId
        );

        if (session is null)
            return;

        _activeSession = session;

        _activeSession.LastActivity =
            DateTimeOffset.UtcNow;
    }


    // =====================================================
    // ADD MESSAGE
    // =====================================================

    public void AddMessage(
        string role,
        string content)
    {
        if (_activeSession is null)
            return;

        var message = new ChatMessage
        {
            SessionId = _activeSession.Id,
            Role = role,
            Content = content,
            CreatedAt = DateTimeOffset.UtcNow
        };

        _activeSession.Messages.Add(message);

        _activeSession.LastActivity =
            DateTimeOffset.UtcNow;

        if (
            _activeSession.Title == "New conversation" &&
            role == "user"
        )
        {
            _activeSession.Title =
                CreateTitle(content);
        }
    }


    // =====================================================
    // CREATE CHAT TITLE
    // =====================================================

    private static string CreateTitle(
        string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            return "New conversation";

        var title = message.Trim();

        if (title.Length > 35)
            title = title[..35] + "...";

        return title;
    }


    // =====================================================
    // RESTORE SESSIONS
    // =====================================================

    public void RestoreSessions(
        List<SupportSession> sessions)
    {
        _sessions.Clear();

        if (sessions.Count == 0)
            return;

        _sessions.AddRange(sessions);

        _activeSession = _sessions[0];
    }


    // =====================================================
    // REMOVE EXPIRED SESSIONS
    // =====================================================

    public void RemoveExpiredSessions()
    {
        var now = DateTimeOffset.UtcNow;

        foreach (var session in _sessions)
        {
            // Existing chats created before ExpiresAt was added
            // receive an expiration date based on StartedAt.
            if (session.ExpiresAt == default)
            {
                session.ExpiresAt =
                    session.StartedAt.AddDays(30);
            }
        }

        _sessions.RemoveAll(
            session => session.ExpiresAt <= now
        );

        if (
            _activeSession is not null &&
            !_sessions.Any(
                x => x.Id == _activeSession.Id
            )
        )
        {
            _activeSession =
                _sessions.FirstOrDefault();
        }
    }


    // =====================================================
    // DELETE ONE SESSION
    // =====================================================

    public void DeleteSession(
        Guid sessionId)
    {
        var session = _sessions.FirstOrDefault(
            x => x.Id == sessionId
        );

        if (session is null)
            return;

        _sessions.Remove(session);

        if (
            _activeSession?.Id == sessionId
        )
        {
            _activeSession =
                _sessions.FirstOrDefault();
        }
    }


    // =====================================================
    // DELETE ALL SESSIONS
    // =====================================================

    public void DeleteAllSessions()
    {
        _sessions.Clear();

        _activeSession = null;
    }


    // =====================================================
    // STORAGE
    // =====================================================

    public List<SupportSession>
        GetSessionsForStorage()
    {
        return _sessions.ToList();
    }
}