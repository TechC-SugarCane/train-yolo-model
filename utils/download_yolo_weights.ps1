<#
.SYNOPSIS
    指定したYOLOモデルの重みファイルをダウンロードします。

.DESCRIPTION
    JSONファイル（weights_map.json）からモデル情報を取得し、指定した重み名のファイルをダウンロードします。
    既にファイルが存在する場合は、再ダウンロードの確認を行います。

.EXAMPLE
    .\utils\download_yolo_weights.ps1 -ModelVersion "yolov10" -WeightNameList s,m,l
.EXAMPLE
    .\utils\download_yolo_weights.ps1 -ModelVersion "yolov10" -WeightNameList x
#>
param(
    [Parameter(Mandatory = $true, HelpMessage = "モデルバージョンを指定してください。 (例: yolov10)")]
    [string]$ModelVersion,

    [Parameter(HelpMessage = "ダウンロードする重み名のリストを指定してください。 (例: s, m, l)")]
    [string[]]$WeightNameList,

    [Parameter(HelpMessage = "再ダウンロードの確認をスキップします。")]
    [switch]$Force
)

# 定数の定義
$WeightDir = ".\weights"
$WeightMapPath = Join-Path -Path $WeightDir -ChildPath 'weights_map.json'

# JSONファイルが存在するか確認
if (-not (Test-Path $WeightMapPath)) {
    Write-Error "JSONファイル '$WeightMapPath' が見つかりません。"
    exit 1
}

# JSONファイルの読み込み（エラーハンドリング付き）
try {
    $WEIGHT_MAP = Get-Content -Raw -Path $WeightMapPath | ConvertFrom-Json -AsHashtable
} catch {
    Write-Error "JSONファイルの読み込みに失敗しました: $_"
    exit 1
}

# パラメータを小文字に統一（配列内の各要素を変換）
$ModelVersion = $ModelVersion.ToLower()
if ($null -ne $WeightNameList) {
    $WeightNameList = $WeightNameList | ForEach-Object { $_.ToLower() }
}

# モデルバージョンの存在確認
if (-not $WEIGHT_MAP.ContainsKey($ModelVersion)) {
    Write-Error "無効なモデルバージョン: $ModelVersion"
    Write-Error "利用可能なモデルバージョン: $($WEIGHT_MAP.Keys -join ', ')"
    exit 1
}

# モデル情報の取得
$ModelInfo = $WEIGHT_MAP[$ModelVersion]
$YOLO_NAME = $ModelInfo.name
$YOLO_WEIGHTS_LIST = $ModelInfo.weights
$YOLO_URL = $ModelInfo.url

# WeightNameListが空の場合は全ての重みをダウンロード
if ($WeightNameList.Length -eq 0) {
    $WeightNameList = $YOLO_WEIGHTS_LIST
}

#-------------------------------------------
# 重みファイルのダウンロード関数
#-------------------------------------------
function Get-YOLOWeight {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ModelName,
        [Parameter(Mandatory = $true)][string]$ModelUrl,
        [Parameter(Mandatory = $true)][string]$ModelVersion,
        [Parameter(Mandatory = $true)][string]$WeightDir,
        [Parameter(Mandatory = $true)][string]$WeightName
    )

    # URLの末尾にスラッシュがなければ追加
    if (-not $ModelUrl.EndsWith('/')) {
        $ModelUrl += '/'
    }

    $FullWeightName = "$ModelName$WeightName.pt"
    $WeightsUrl     = "$ModelUrl$FullWeightName"
    $SaveDir        = Join-Path -Path $WeightDir -ChildPath $ModelVersion
    $WeightsPath    = Join-Path -Path $SaveDir -ChildPath $FullWeightName
    $WeightsPathTmp = "$WeightsPath.tmp"

    # 既に重みファイルが存在する場合の確認
    if (-not $Force -and (Test-Path $WeightsPath)) {
        Write-Host "重みファイル '$FullWeightName' は既に存在します。"
        $response = Read-Host "再ダウンロードしますか？ (y/n) [n]"
        if ($response -notin @('y', 'Y')) {
            return
        }
        Remove-Item $WeightsPath -Force
    }

    # 保存先ディレクトリが存在しない場合は作成
    if (-not (Test-Path $SaveDir)) {
        New-Item -ItemType Directory -Path $SaveDir | Out-Null
    }

    Write-Host "重みファイルをダウンロード中: $WeightsUrl -> $WeightsPath"
    try {
        Invoke-WebRequest -Uri $WeightsUrl -OutFile $WeightsPathTmp -ErrorAction Stop
        Move-Item -Path $WeightsPathTmp -Destination $WeightsPath -Force
        Write-Host "ダウンロード完了: $WeightsPath"
    }
    catch {
        Write-Error "ダウンロード中にエラーが発生しました: $_"
    }
    finally {
        # 一時ファイルが存在する場合は削除
        if (Test-Path $WeightsPathTmp) {
            Remove-Item $WeightsPathTmp -Force
        }
    }
}

#-------------------------------------------
# 重み名の妥当性検証関数
#-------------------------------------------
function Test-ValidWeightName {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string[]]$AvailableWeightNames,
        [Parameter(Mandatory = $true)][string]$WeightName
    )
    if ($AvailableWeightNames -contains $WeightName) {
        Write-Host "有効な重み名を確認: $WeightName"
        return $true
    } else {
        Write-Error "無効な重み名: $WeightName. 利用可能な重み名: $($AvailableWeightNames -join ', ')"
        return $false
    }
}

#-------------------------------------------
# メイン処理：指定された重み名に対してダウンロード実行
#-------------------------------------------
Write-Host "モデルバージョン: $ModelVersion"
foreach ($weightName in $WeightNameList) {
    Write-Host "------------------------------------------------------------"
    if (Test-ValidWeightName -AvailableWeightNames $YOLO_WEIGHTS_LIST -WeightName $weightName) {
        Get-YOLOWeight -WeightDir $WeightDir `
                        -ModelName $YOLO_NAME `
                        -ModelUrl $YOLO_URL `
                        -ModelVersion $ModelVersion `
                        -WeightName $weightName
    }
}
