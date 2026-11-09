param(
  [string]$BaseUrl = "http://127.0.0.1:8001",
  [string]$Output = "$env:GITHUB_WORKSPACE\deployment-results\requests.csv",
  [string]$PhaseFile = "$env:GITHUB_WORKSPACE\deployment-results\phase.txt",
  [string]$StopFile = "$env:GITHUB_WORKSPACE\deployment-results\stop"
)

New-Item -ItemType Directory -Force (Split-Path $Output) | Out-Null
"stage,sequence,start,end,http,status,version,response_ms,result" |
  Set-Content $Output -Encoding UTF8

$i = 0

while (-not (Test-Path $StopFile)) {
  $phase = if (Test-Path $PhaseFile) {
    (Get-Content $PhaseFile -Raw).Trim()
  } else {
    "Before"
  }

  $i++
  $start = Get-Date
  $status = "ERROR"
  $http = "0"
  $version = ""
  $result = "FAIL"

  try {
    $response = Invoke-WebRequest `
      "$BaseUrl/api/health" `
      -UseBasicParsing `
      -TimeoutSec 5

    $body = $response.Content | ConvertFrom-Json
    $http = $response.StatusCode
    $status = "OK"
    $version = $body.model_version
    $result = if ($version -eq "2.0") { "PASS" } else { "WRONG_VERSION" }
  }
  catch {
    $result = "CONNECTION_ERROR"
  }

  $end = Get-Date
  $ms = [math]::Round(($end - $start).TotalMilliseconds, 2)

  "$phase,$i,$($start.ToUniversalTime().ToString('o')),$($end.ToUniversalTime().ToString('o')),$http,$status,$version,$ms,$result" |
    Add-Content $Output -Encoding UTF8

  Start-Sleep -Seconds 1
}
