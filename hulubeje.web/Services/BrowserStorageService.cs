using System.Text.Json;
using Microsoft.JSInterop;

namespace HuluBeje.Web.Services;

public class BrowserStorageService
{
    private readonly IJSRuntime _js;

    public BrowserStorageService(IJSRuntime js)
    {
        _js = js;
    }

    public async Task SetAsync<T>(string key, T value)
    {
        var json = JsonSerializer.Serialize(value);

        await _js.InvokeVoidAsync(
            "localStorage.setItem",
            key,
            json);
    }

    public async Task<T?> GetAsync<T>(string key)
    {
        var json = await _js.InvokeAsync<string?>(
            "localStorage.getItem",
            key);

        if (string.IsNullOrWhiteSpace(json))
            return default;

        return JsonSerializer.Deserialize<T>(json);
    }

    public async Task RemoveAsync(string key)
    {
        await _js.InvokeVoidAsync(
            "localStorage.removeItem",
            key);
    }
}