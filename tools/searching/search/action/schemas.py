from pydantic import BaseModel, field_validator


class SearchParams(BaseModel):
    pubAlias: str | None = None
    fixedregioncode: str | None = None
    isUseHints: str | None = None
    fstring: str | None = None
    sortby: str | None = None
    status: str | None = None
    dataformat: str | None = None
    pubdivid: int | None = None
    pubId: int | None = None
    page: int | None = None

    class Config:
        extra = "allow"


class SearchItem(BaseModel):
    id: str | None = None
    moduleId: str | None = None
    url: str | None = None
    docName: str | None = None
    snippet: str | None = None
    anchor: str | None = None

    # locale: str | None = None # TODO: implement
    position: int | None = None
    score: float | None = None
    isEtalon: bool | None = None
    isPopular: bool | None = None

    # TODO: implement snippet and docName cleaning

    @field_validator("id", "moduleId", mode="before")
    @classmethod
    def convert_to_string(cls, v):
        if v is not None:
            return str(v)
        return v

    class Config:
        extra = "allow"


class SearchResult(BaseModel):
    item: SearchItem
    document: dict
    error: str | None = None
